import SwiftUI
import CoreML
import CoreMedia
import CoreVideo
import CryptoKit
import ImageIO
import UniformTypeIdentifiers
import Darwin

enum OwnedDeviceMode: String, CaseIterable, Identifiable, Sendable {
    case cpu = "CPU only"
    case neuralEngine = "CPU + Neural Engine"
    case gpu = "CPU + GPU"
    case automatic = "Automatic"
    var id: String { rawValue }
    var units: MLComputeUnits {
        switch self {
        case .cpu: return .cpuOnly
        case .neuralEngine: return .cpuAndNeuralEngine
        case .gpu: return .cpuAndGPU
        case .automatic: return .all
        }
    }
}

private struct OwnedFixture: Decodable {
    struct TensorFile: Decodable {
        let path: String
        let sha256: String
        let shape: [Int]
    }
    struct Frame: Decodable {
        let index: Int
        let ptsNumerator: Int64
        let ptsDenominator: Int32
        let imageSHA256: String
        let inputPath: String
        let inputSHA256: String
        let previewPath: String
        let tensors: [String: TensorFile]
    }
    let schema: String
    let contract: String
    let precisionPolicy: String
    let graphRevision: String?
    let upstream: String
    let checkpointSHA256: String
    let sourceModelsManifestSHA256: String
    let sourceReportSHA256: String
    let pointNormalizedTopLeft: [Double]
    let referenceComputeUnits: String
    let modelFiles: [String: String]
    let frames: [Frame]
}

struct OwnedTensorComparison: Codable, Sendable {
    let maximumAbsoluteError: Double
    let cosineSimilarity: Double
}

struct OwnedFrameComparison: Codable, Sendable {
    let index: Int
    let ptsNumerator: Int64
    let ptsDenominator: Int32
    let sourceImageSHA256: String
    let inputBGRASHA256: String
    let modelMilliseconds: [String: Double]
    let predictionAndStateMillisecondsDiagnosticOnly: Double
    let maskIoUAgainstMacCoreML: Double
    let sameObjectPresence: Bool
    let foregroundFraction: Double
    let outputs: [String: OwnedTensorComparison]
    let state: OwnedStateSummary
    let passed: Bool
}

struct OwnedDeviceReport: Encodable, Sendable {
    let schema = "bytewave.temporal-device-comparison.v1"
    let contract = OwnedTemporalContract.id
    let precisionPolicy = OwnedTemporalContract.precision
    let graphRevision = OwnedTemporalContract.graphRevision
    let generatedAt = Date()
    let hardware: String
    let operatingSystem = ProcessInfo.processInfo.operatingSystemVersionString
    let isIOSAppOnMac = ProcessInfo.processInfo.isiOSAppOnMac
    let mode: String
    let reference = "Mac Core ML CPU replay of the passed original-PyTorch comparison fixture"
    let scope = "20 exact-input sequential predictions with bounded Swift state. Low mask, pointer and memory cosine >= 0.99; low-mask IoU >= 0.95; matching object presence; all model outputs finite. High mask is checked for shape/finiteness, not compared numerically. No video decoding, sustained playback FPS, or measured hardware utilization."
    let timingScope = "Model-call wall time excludes input copies, tensor checks, state assembly, reference comparison, checkpoint file I/O and display. Prediction-and-state time includes input copies/checks/state and pre-prediction checkpoint writes. First frame is cold; compile/load is separate. Debug-build timings are diagnostic."
    var fixtureSHA256: String?
    var sourceModelsManifestSHA256: String?
    var sourceReportSHA256: String?
    var pointNormalizedTopLeft: [Double]?
    var verifiedModelFileCount = 0
    var compileAndLoadMilliseconds: [String: Double] = [:]
    var frames: [OwnedFrameComparison] = []
    var passed = false
    var completed = false
    var lastCheckpoint: String?
    var activeFrameIndex: Int?
    var activeComponent: String?
    var cancelled = false
    var failedStage: String?
    var error: String?
    var thermalStateAtStart: String
    var thermalStateAtEnd: String?
}

struct OwnedProbeUpdate: Sendable {
    let message: String
    var previewPNG: Data? = nil
    var overlayPNG: Data? = nil
}

actor OwnedTemporalProbeRunner {
    private var running = false

    func run(root: URL, mode: OwnedDeviceMode, hardware: String,
             progress: @Sendable (OwnedProbeUpdate) async -> Void) async -> OwnedDeviceReport {
        var report = OwnedDeviceReport(hardware: hardware, mode: mode.rawValue, thermalStateAtStart: thermal())
        guard !running else {
            report.error = "A temporal comparison is already running."
            return report
        }
        running = true
        var models: [String: MLModel] = [:]
        var compiledURLs: [URL] = []
        var stage = "fixture validation"
        var checkpointURL: URL?
        defer {
            models.removeAll()
            for url in compiledURLs { try? FileManager.default.removeItem(at: url) }
            running = false
        }
        do {
            let documents = try FileManager.default.url(for: .documentDirectory, in: .userDomainMask,
                                                        appropriateFor: nil, create: true)
            let checkpoint = documents.appendingPathComponent("temporal-last-run.json")
            checkpointURL = checkpoint
            report.lastCheckpoint = stage
            try saveCheckpoint(report, to: checkpoint)
            let data = try Data(contentsOf: root.appendingPathComponent("fixture.json"))
            let fixture = try JSONDecoder().decode(OwnedFixture.self, from: data)
            report.fixtureSHA256 = digest(data)
            report.sourceModelsManifestSHA256 = fixture.sourceModelsManifestSHA256
            report.sourceReportSHA256 = fixture.sourceReportSHA256
            report.pointNormalizedTopLeft = fixture.pointNormalizedTopLeft
            guard fixture.schema == "bytewave.temporal-device-fixture.v1",
                  fixture.contract == OwnedTemporalContract.id,
                  fixture.precisionPolicy == OwnedTemporalContract.precision,
                  fixture.graphRevision == OwnedTemporalContract.graphRevision,
                  fixture.upstream == OwnedTemporalContract.upstream,
                  fixture.checkpointSHA256 == OwnedTemporalContract.checkpoint,
                  fixture.referenceComputeUnits == "CPU_ONLY", fixture.frames.count == 20,
                  fixture.pointNormalizedTopLeft.count == 2,
                  fixture.pointNormalizedTopLeft.allSatisfy({ $0.isFinite && (0...1).contains($0) }),
                  !fixture.modelFiles.isEmpty else {
                throw OwnedTemporalError.invalid("Missing or incompatible device fixture. Run the Mac preparation command first.")
            }
            await progress(OwnedProbeUpdate(message: "Verifying model files…"))
            for (path, hash) in fixture.modelFiles.sorted(by: { $0.key < $1.key }) {
                try Task.checkCancellation()
                _ = try verifiedData(root, path, hash)
                report.verifiedModelFileCount += 1
            }
            for component in OwnedTemporalContract.components {
                stage = "compile/load \(component)"
                try Task.checkCancellation()
                report.activeComponent = component
                report.lastCheckpoint = stage
                try saveCheckpoint(report, to: checkpoint)
                await progress(OwnedProbeUpdate(message: "Loading \(component)…"))
                let start = ProcessInfo.processInfo.systemUptime
                let package = root.appendingPathComponent("models/BWTemporal\(component).mlpackage")
                let compiled = try await MLModel.compileModel(at: package)
                compiledURLs.append(compiled)
                let configuration = MLModelConfiguration()
                configuration.computeUnits = mode.units
                let model = try await MLModel.load(contentsOf: compiled, configuration: configuration)
                try OwnedTemporalContract.validate(model, component: component)
                models[component] = model
                report.compileAndLoadMilliseconds[component] = (ProcessInfo.processInfo.systemUptime - start) * 1000
            }
            // The session is local to this run. No await occurs inside predict;
            // a cancellation/new run cannot commit an old frame into new state.
            let session = OwnedTemporalSession(models: models)
            defer { session.reset() }
            for (index, frame) in fixture.frames.enumerated() {
                stage = "frame \(index + 1)"
                report.activeFrameIndex = index
                report.activeComponent = nil
                report.lastCheckpoint = "Preparing \(stage)"
                try saveCheckpoint(report, to: checkpoint)
                try Task.checkCancellation()
                guard frame.index == index, frame.ptsDenominator > 0 else {
                    throw OwnedTemporalError.invalid("Invalid fixture frame order or timestamp.")
                }
                let result = try autoreleasepool { () throws -> (OwnedFrameComparison, Data, Data) in
                    let bytes = try verifiedData(root, frame.inputPath, frame.inputSHA256)
                    let buffer = try pixelBuffer(bytes)
                    let timestamp = CMTime(value: frame.ptsNumerator, timescale: frame.ptsDenominator)
                    let start = ProcessInfo.processInfo.systemUptime
                    let prediction = try session.predict(image: buffer, at: timestamp,
                                                         initialPoint: fixture.pointNormalizedTopLeft) { component in
                        stage = "frame \(index + 1), \(component)"
                        report.activeComponent = component
                        report.lastCheckpoint = "Before prediction: \(stage)"
                        try saveCheckpoint(report, to: checkpoint)
                        print("[OwnedTemporal] \(mode.rawValue): \(stage), beginning prediction")
                        fflush(stdout)
                    }
                    let elapsed = (ProcessInfo.processInfo.systemUptime - start) * 1000
                    stage = "frame \(index + 1), reference comparison"
                    let expectedNames = Set(["low_res_mask", "best_iou", "object_pointer", "object_score",
                                             "memory_features", "memory_positions"])
                    guard Set(frame.tensors.keys) == expectedNames else {
                        throw OwnedTemporalError.invalid("Reference tensor set is incomplete.")
                    }
                    var references: [String: OwnedTensor] = [:]
                    var comparisons: [String: OwnedTensorComparison] = [:]
                    for name in expectedNames.sorted() {
                        guard let file = frame.tensors[name], file.shape == OwnedTemporalContract.shapes[name],
                              let actual = prediction.tensors[name] else {
                            throw OwnedTemporalError.invalid("Missing comparison tensor \(name).")
                        }
                        let reference = try readTensor(verifiedData(root, file.path, file.sha256), shape: file.shape)
                        references[name] = reference
                        comparisons[name] = compare(actual.values, reference.values)
                    }
                    guard let actualMask = prediction.tensors["low_res_mask"], let expectedMask = references["low_res_mask"],
                          let actualScore = prediction.tensors["object_score"]?.values.first,
                          let expectedScore = references["object_score"]?.values.first else {
                        throw OwnedTemporalError.invalid("Missing mask or presence score.")
                    }
                    var intersection = 0
                    var union = 0
                    var foreground = 0
                    for i in actualMask.values.indices {
                        let actual = actualMask.values[i] > 0
                        let expected = expectedMask.values[i] > 0
                        if actual { foreground += 1 }
                        if actual && expected { intersection += 1 }
                        if actual || expected { union += 1 }
                    }
                    if index == 0 && !expectedMask.values.contains(where: { $0 > 0 }) {
                        throw OwnedTemporalError.invalid("Empty reference subject on frame 1.")
                    }
                    let iou = union == 0 ? 1 : Double(intersection) / Double(union)
                    let presence = (actualScore > 0) == (expectedScore > 0)
                    let cosinePassed = ["low_res_mask", "object_pointer", "memory_features", "memory_positions"]
                        .allSatisfy { (comparisons[$0]?.cosineSimilarity ?? -1) >= 0.99 }
                    let state = prediction.state
                    let statePassed = state.acceptedFrames == index + 1 && state.spatialEntries == min(index + 1, 7)
                        && state.pointerEntries == min(index + 1, 16)
                    let row = OwnedFrameComparison(index: index, ptsNumerator: frame.ptsNumerator,
                        ptsDenominator: frame.ptsDenominator, sourceImageSHA256: frame.imageSHA256,
                        inputBGRASHA256: frame.inputSHA256, modelMilliseconds: prediction.modelMilliseconds,
                        predictionAndStateMillisecondsDiagnosticOnly: elapsed, maskIoUAgainstMacCoreML: iou,
                        sameObjectPresence: presence, foregroundFraction: Double(foreground) / Double(actualMask.values.count),
                        outputs: comparisons, state: state, passed: iou >= 0.95 && presence && cosinePassed && statePassed)
                    let preview = try Data(contentsOf: safeURL(root, frame.previewPath))
                    return (row, preview, try maskPNG(actualMask.values))
                }
                report.frames.append(result.0)
                report.activeComponent = nil
                report.lastCheckpoint = "Completed frame \(index + 1) comparison"
                try saveCheckpoint(report, to: checkpoint)
                await progress(OwnedProbeUpdate(message: "Frame \(index + 1)/20: \(result.0.passed ? "passed" : "FAILED")",
                                               previewPNG: result.1, overlayPNG: result.2))
                try Task.checkCancellation()
                guard result.0.passed else {
                    throw OwnedTemporalError.invalid("Device comparison failed at frame \(index + 1). Share the report.")
                }
            }
            report.passed = report.frames.count == 20 && report.frames.allSatisfy(\.passed)
        } catch {
            report.cancelled = error is CancellationError
            report.failedStage = stage
            report.error = error.localizedDescription
        }
        report.completed = true
        report.thermalStateAtEnd = thermal()
        if let checkpointURL {
            do { try saveCheckpoint(report, to: checkpointURL) }
            catch { print("[OwnedTemporal] Final checkpoint save failed: \(error.localizedDescription)") }
        }
        return report
    }

    private func saveCheckpoint(_ report: OwnedDeviceReport, to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(report).write(to: url, options: .atomic)
    }

    private func safeURL(_ root: URL, _ relative: String) throws -> URL {
        let base = root.standardizedFileURL.resolvingSymlinksInPath()
        let url = base.appendingPathComponent(relative).standardizedFileURL.resolvingSymlinksInPath()
        guard !relative.hasPrefix("/"), url.path.hasPrefix(base.path + "/") else {
            throw OwnedTemporalError.invalid("Invalid fixture path.")
        }
        return url
    }

    private func digest(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private func verifiedData(_ root: URL, _ relative: String, _ expected: String) throws -> Data {
        let data = try Data(contentsOf: safeURL(root, relative), options: .mappedIfSafe)
        guard digest(data) == expected else { throw OwnedTemporalError.invalid("Fixture hash mismatch: \(relative).") }
        return data
    }

    private func readTensor(_ data: Data, shape: [Int]) throws -> OwnedTensor {
        let count = shape.reduce(1, *)
        guard data.count == count * 4 else { throw OwnedTemporalError.invalid("Truncated reference tensor.") }
        let values = data.withUnsafeBytes { bytes in
            (0..<count).map { Float(bitPattern: UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: $0 * 4, as: UInt32.self))) }
        }
        return try OwnedTensor(shape: shape, values: values)
    }

    private func pixelBuffer(_ bytes: Data) throws -> CVPixelBuffer {
        guard bytes.count == 1024 * 1024 * 4 else { throw OwnedTemporalError.invalid("Invalid BGRA input size.") }
        var optional: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, 1024, 1024, kCVPixelFormatType_32BGRA,
                                        [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &optional)
        guard status == kCVReturnSuccess, let buffer = optional else {
            throw OwnedTemporalError.invalid("Pixel buffer allocation failed (\(status)).")
        }
        guard CVPixelBufferLockBaseAddress(buffer, []) == kCVReturnSuccess else {
            throw OwnedTemporalError.invalid("Pixel buffer lock failed.")
        }
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { throw OwnedTemporalError.invalid("No pixel buffer storage.") }
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        bytes.withUnsafeBytes { source in
            guard let address = source.baseAddress else { return }
            for y in 0..<1024 {
                base.advanced(by: y * stride).copyMemory(from: address.advanced(by: y * 1024 * 4), byteCount: 1024 * 4)
            }
        }
        return buffer
    }

    private func compare(_ actual: [Float], _ expected: [Float]) -> OwnedTensorComparison {
        var maximum = 0.0, dot = 0.0, normA = 0.0, normB = 0.0
        for i in actual.indices {
            let a = Double(actual[i]), b = Double(expected[i])
            maximum = max(maximum, abs(a - b))
            dot += a * b
            normA += a * a
            normB += b * b
        }
        let denominator = sqrt(normA) * sqrt(normB)
        let cosine = denominator > 0 ? dot / denominator : (maximum == 0 ? 1 : 0)
        return OwnedTensorComparison(maximumAbsoluteError: maximum, cosineSimilarity: cosine)
    }

    private func maskPNG(_ values: [Float]) throws -> Data {
        var bytes = [UInt8](repeating: 0, count: 256 * 256 * 4)
        for i in values.indices where values[i] > 0 {
            bytes[i * 4] = 90
            bytes[i * 4 + 1] = 160
            bytes[i * 4 + 2] = 255
            bytes[i * 4 + 3] = 255
        }
        guard let provider = CGDataProvider(data: Data(bytes) as CFData),
              let image = CGImage(width: 256, height: 256, bitsPerComponent: 8, bitsPerPixel: 32,
                  bytesPerRow: 256 * 4, space: CGColorSpaceCreateDeviceRGB(),
                  bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                  provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent) else {
            throw OwnedTemporalError.invalid("Could not draw the mask.")
        }
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(data as CFMutableData, UTType.png.identifier as CFString, 1, nil) else {
            throw OwnedTemporalError.invalid("Could not encode mask PNG.")
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { throw OwnedTemporalError.invalid("PNG encoding failed.") }
        return data as Data
    }

    private func thermal() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }
}

@MainActor
struct OwnedTemporalProbeView: View {
    @State private var mode: OwnedDeviceMode = .cpu
    @State private var running = false
    @State private var status = "Run the prepared 20-frame subject-tracking comparison."
    @State private var preview: UIImage?
    @State private var overlay: UIImage?
    @State private var showMask = true
    @State private var reportURL: URL?
    @State private var job: Task<Void, Never>?
    @State private var runner = OwnedTemporalProbeRunner()

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Picker("Compute devices", selection: $mode) {
                    ForEach(OwnedDeviceMode.allCases) { Text($0.rawValue).tag($0) }
                }.disabled(running)
                Button(running ? "Comparing…" : "Run 20-frame comparison", action: start)
                    .buttonStyle(.borderedProminent).disabled(running)
                if running { ProgressView() }
                Text(status).textSelection(.enabled).accessibilityAddTraits(.updatesFrequently)
                if let preview {
                    ZStack {
                        Image(uiImage: preview).resizable().aspectRatio(contentMode: .fit)
                        if showMask, let overlay {
                            Image(uiImage: overlay).resizable().interpolation(.none).opacity(0.45)
                        }
                    }.aspectRatio(1, contentMode: .fit)
                    Toggle("Show subject mask", isOn: $showMask)
                }
                if let reportURL { ShareLink("Share temporal report", item: reportURL) }
                Text("Tracks one subject through 20 prepared frames, keeping a bounded memory. Each run starts fresh. This checks agreement with your Mac; it is not a playback-speed test.")
                    .font(.footnote).foregroundStyle(.secondary)
            }.padding()
        }
        .navigationTitle("Temporal comparison")
        .onAppear {
            guard !running, reportURL == nil,
                  let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { return }
            let previous = documents.appendingPathComponent("temporal-last-run.json")
            if FileManager.default.fileExists(atPath: previous.path) {
                reportURL = previous
                status = "The previous run's report is available, including its last saved step if interrupted."
            }
        }
        .onChange(of: mode) { _, _ in
            reportURL = nil
            preview = nil
            overlay = nil
            status = "Run the comparison in this mode."
        }
        .onDisappear { job?.cancel() }
    }

    private func start() {
        #if targetEnvironment(simulator)
        status = "Run the temporal comparison on your physical iPhone."
        return
        #else
        guard !running, let root = Bundle.main.resourceURL?.appendingPathComponent("DeviceValidationData/baseline") else { return }
        guard FileManager.default.fileExists(atPath: root.appendingPathComponent("fixture.json").path) else {
            status = "The device fixture is missing. Run the Mac preparation command, then rebuild."
            return
        }
        running = true
        reportURL = nil
        preview = nil
        overlay = nil
        let selectedMode = mode
        let hardware = hardwareIdentifier()
        job = Task {
            defer { running = false }
            let report = await runner.run(root: root, mode: selectedMode, hardware: hardware) { update in
                await MainActor.run {
                    guard !Task.isCancelled else { return }
                    status = update.message
                    if let data = update.previewPNG { preview = UIImage(data: data) }
                    if let data = update.overlayPNG { overlay = UIImage(data: data) }
                }
            }
            guard !Task.isCancelled else { return }
            do {
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                encoder.dateEncodingStrategy = .iso8601
                let directory = try FileManager.default.url(for: .documentDirectory, in: .userDomainMask,
                                                            appropriateFor: nil, create: true)
                let url = directory.appendingPathComponent("temporal-\(UUID().uuidString).json")
                try encoder.encode(report).write(to: url, options: .atomic)
                reportURL = url
                status = report.passed ? "All 20 frames passed. Share the temporal report."
                    : "Comparison stopped: \(report.error ?? "unknown error"). Share the temporal report."
            } catch { status = "Could not save the report: \(error.localizedDescription)" }
        }
        #endif
    }

    private func hardwareIdentifier() -> String {
        var size: size_t = 0
        guard sysctlbyname("hw.machine", nil, &size, nil, 0) == 0, size > 0 else { return "Unknown" }
        var bytes = [CChar](repeating: 0, count: size)
        guard sysctlbyname("hw.machine", &bytes, &size, nil, 0) == 0 else { return "Unknown" }
        return String(cString: bytes)
    }
}
