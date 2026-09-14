import SwiftUI
import PhotosUI
import CoreTransferable
import UniformTypeIdentifiers
import AVFoundation
import CoreML
import CoreImage
import ImageIO
import Darwin

// This path deliberately stops before the undocumented propagator interface.
// See Audit/temporal-contract.md before adding any temporal state assembly.
private enum FrameProbeError: LocalizedError {
    case invalid(String)
    var errorDescription: String? {
        switch self { case .invalid(let message): return message }
    }
}

struct ProbeMovie: Transferable {
    let url: URL

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(importedContentType: .movie) { received in
            let directory = FileManager.default.temporaryDirectory
                .appendingPathComponent("ByteWaveFrameImport", isDirectory: true)
                .appendingPathComponent(UUID().uuidString, isDirectory: true)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let url = directory.appendingPathComponent("clip")
                .appendingPathExtension(received.file.pathExtension)
            do {
                try FileManager.default.copyItem(at: received.file, to: url)
                return ProbeMovie(url: url)
            } catch {
                try? FileManager.default.removeItem(at: directory)
                throw error
            }
        }
    }
}

struct FramePreview: Sendable {
    let png: Data
    let width: Int
    let height: Int
    let actualTimeValue: Int64
    let actualTimeScale: Int32
}

struct PredictionTiming: Codable, Sendable {
    let imageEncoderMilliseconds: Double
    let initializerMilliseconds: Double
    let memoryEncoderMilliseconds: Double
}

struct TensorCheck: Codable, Sendable {
    let name: String
    let shape: [Int]
    let minimum: Double
    let maximum: Double
}

struct FirstFrameReport: Encodable, Sendable {
    let schema = "bytewave.segmentation-first-frame.v1"
    let modelRevision = "6bfdd4765e42508c7707566fff52e65add8b8e3a"
    let generatedAt = Date()
    let hardware: String
    let operatingSystem = ProcessInfo.processInfo.operatingSystemVersionString
    let mode: String
    let previewWidth: Int
    let previewHeight: Int
    let actualTimeValue: Int64
    let actualTimeScale: Int32
    let promptNormalizedTopLeft: [Double]
    let promptModelXY: [Double]
    let positivePointLabel = 1
    let imagePreparation = "Display-oriented frame, long edge at most 1024; stretch to 1024x1024 RGB via BGRA pixel buffer. No extra mean/std normalization."
    let measurementScope = "One untimed warm-up and three measured repetitions of image encoder, initializer, and initial memory encoder on the SAME frame and point. Per-call wall time excludes compile/load, preprocessing, diagnostics and mask rendering. No propagator, video FPS, sustained performance or runtime hardware-utilization measurement."
    var compileAndLoadMilliseconds: Double?
    var preprocessingMilliseconds: Double?
    var samples: [PredictionTiming] = []
    var tensorChecks: [TensorCheck] = []
    var foregroundFraction: Double?
    var bestIoU: Double?
    var objectScore: Double?
    var thermalStateAtStart: String
    var thermalStateAtEnd: String?
    var succeeded = false
    var failedStage: String?
    var error: String?
}

struct FirstFrameResult: Sendable {
    let report: FirstFrameReport
    let overlayPNG: Data?
}

actor FirstFrameRunner {
    private var frame: CGImage?
    private var preview: FramePreview?
    private let context = CIContext(options: [.cacheIntermediates: false])

    func reset() {
        frame = nil
        preview = nil
        context.clearCaches()
    }

    func open(url: URL) async throws -> FramePreview {
        reset()
        let asset = AVURLAsset(url: url)
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: 1024, height: 1024)
        // Record the actual returned timestamp; do not assume frame zero is at t=0.
        let result = try await generator.image(at: .zero)
        try Task.checkCancellation()
        guard result.actualTime.isNumeric else {
            throw FrameProbeError.invalid("The video returned an invalid frame timestamp.")
        }
        let info = FramePreview(png: try png(result.image), width: result.image.width,
                                height: result.image.height,
                                actualTimeValue: result.actualTime.value,
                                actualTimeScale: result.actualTime.timescale)
        frame = result.image
        preview = info
        return info
    }

    func run(point: CGPoint, mode: ProbeMode, modelsDirectory: URL, hardware: String,
             progress: @Sendable (String) async -> Void) async throws -> FirstFrameResult {
        guard let frame, let preview else { throw FrameProbeError.invalid("Choose a video first.") }
        guard point.x.isFinite, point.y.isFinite,
              (0...1).contains(point.x), (0...1).contains(point.y) else {
            throw FrameProbeError.invalid("Tap inside the video frame.")
        }
        let x = min(Double(point.x) * 1024, 1023)
        let y = min(Double(point.y) * 1024, 1023)
        var report = FirstFrameReport(hardware: hardware, mode: mode.rawValue,
                                      previewWidth: preview.width, previewHeight: preview.height,
                                      actualTimeValue: preview.actualTimeValue,
                                      actualTimeScale: preview.actualTimeScale,
                                      promptNormalizedTopLeft: [Double(point.x), Double(point.y)],
                                      promptModelXY: [x, y],
                                      thermalStateAtStart: thermalState())
        var compiledURLs: [URL] = []
        var models: [String: MLModel] = [:]
        var stage = "loading"
        defer {
            models.removeAll()
            for url in compiledURLs { try? FileManager.default.removeItem(at: url) }
            context.clearCaches()
        }
        do {
            let loadStart = now()
            for name in ["ImageEncoder", "Initializer", "MemoryEncoder"] {
                stage = "compile/load \(name)"
                try Task.checkCancellation()
                await progress("Loading \(name)…")
                let package = modelsDirectory.appendingPathComponent("EdgeTAMVideo\(name).mlpackage")
                let compiled = try MLModel.compileModel(at: package)
                compiledURLs.append(compiled)
                let config = MLModelConfiguration()
                config.computeUnits = mode.computeUnits
                models[name] = try await MLModel.load(contentsOf: compiled, configuration: config)
            }
            report.compileAndLoadMilliseconds = milliseconds(since: loadStart)
            guard let imageModel = models["ImageEncoder"], let initializer = models["Initializer"],
                  let memoryModel = models["MemoryEncoder"] else {
                throw FrameProbeError.invalid("A model did not load.")
            }
            let preparationStart = now()
            stage = "image and point preparation"
            let buffer = try pixelBuffer(frame)
            let imageInput = try MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: buffer)])
            let coords = try MLMultiArray(shape: [1, 1, 2], dataType: .float16)
            coords[[0, 0, 0]] = NSNumber(value: x)
            coords[[0, 0, 1]] = NSNumber(value: y)
            let labels = try MLMultiArray(shape: [1, 1], dataType: .int32)
            labels[[0, 0]] = 1
            report.preprocessingMilliseconds = milliseconds(since: preparationStart)

            var overlay: Data?
            for pass in 0...3 {
                try Task.checkCancellation()
                await progress(pass == 0 ? "Warming up predictions…" : "Predicting sample \(pass) of 3…")
                // Release intermediate tensors after each repetition; retain no per-frame bank.
                let result = try autoreleasepool { () throws -> (PredictionTiming, Data?, [TensorCheck], Double?, Double?, Double?) in
                    try Task.checkCancellation()
                    stage = "image encoder, pass \(pass)"
                    let imageStart = now()
                    let features = try imageModel.prediction(from: imageInput)
                    let imageMS = milliseconds(since: imageStart)
                    let raw = try tensor(features, "raw_vision_features", [1, 256, 64, 64])
                    let initial = try tensor(features, "initial_vision_features", [1, 256, 64, 64])
                    let high0 = try tensor(features, "high_res_feature_0", [1, 32, 256, 256])
                    let high1 = try tensor(features, "high_res_feature_1", [1, 64, 128, 128])
                    let initialInput = try MLDictionaryFeatureProvider(dictionary: [
                        "initial_vision_features": initial, "high_res_feature_0": high0,
                        "high_res_feature_1": high1, "point_coords": coords, "point_labels": labels
                    ])
                    try Task.checkCancellation()
                    stage = "initializer, pass \(pass)"
                    let initialStart = now()
                    let masks = try initializer.prediction(from: initialInput)
                    let initialMS = milliseconds(since: initialStart)
                    let lowMask = try tensor(masks, "low_res_mask", [1, 1, 256, 256])
                    let highMask = try tensor(masks, "high_res_mask", [1, 1, 1024, 1024])
                    let score = try tensor(masks, "object_score", [1, 1])
                    let iou = try tensor(masks, "best_iou", [1])
                    let pointer = try tensor(masks, "object_pointer", [1, 256])
                    // Pass original high-resolution logits and score unchanged. The memory
                    // encoder owns its mask transformation; display thresholding is separate.
                    let memoryInput = try MLDictionaryFeatureProvider(dictionary: [
                        "raw_vision_features": raw, "high_res_mask": highMask, "object_score": score
                    ])
                    try Task.checkCancellation()
                    stage = "memory encoder, pass \(pass)"
                    let memoryStart = now()
                    let memory = try memoryModel.prediction(from: memoryInput)
                    let memoryMS = milliseconds(since: memoryStart)
                    let memoryFeatures = try tensor(memory, "memory_features", [1, 512, 64])
                    let memoryPositions = try tensor(memory, "memory_positions", [1, 512, 64])
                    let temporalPositions = try tensor(memory, "temporal_positions", [7, 64])
                    let timing = PredictionTiming(imageEncoderMilliseconds: imageMS,
                                                  initializerMilliseconds: initialMS,
                                                  memoryEncoderMilliseconds: memoryMS)
                    guard pass == 3 else { return (timing, nil, [], nil, nil, nil) }
                    try Task.checkCancellation()
                    stage = "tensor validation and mask rendering"
                    let checks = try [
                        check(lowMask, name: "low_res_mask"),
                        check(highMask, name: "high_res_mask"),
                        check(score, name: "object_score"), check(iou, name: "best_iou"),
                        check(pointer, name: "object_pointer"),
                        check(memoryFeatures, name: "memory_features"),
                        check(memoryPositions, name: "memory_positions"),
                        check(temporalPositions, name: "temporal_positions")
                    ]
                    let rendered = try maskPNG(lowMask)
                    return (timing, rendered.0, checks, rendered.1,
                            iou[[0]].doubleValue, score[[0, 0]].doubleValue)
                }
                if pass > 0 { report.samples.append(result.0) }
                if pass == 3 {
                    overlay = result.1
                    report.tensorChecks = result.2
                    report.foregroundFraction = result.3
                    report.bestIoU = result.4
                    report.objectScore = result.5
                }
            }
            report.succeeded = true
            report.thermalStateAtEnd = thermalState()
            return FirstFrameResult(report: report, overlayPNG: overlay)
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            let error = error as NSError
            report.failedStage = stage
            report.error = "\(error.domain) [\(error.code)]: \(error.localizedDescription)"
            report.thermalStateAtEnd = thermalState()
            return FirstFrameResult(report: report, overlayPNG: nil)
        }
    }

    private func tensor(_ output: MLFeatureProvider, _ name: String, _ shape: [Int]) throws -> MLMultiArray {
        guard let array = output.featureValue(for: name)?.multiArrayValue,
              array.shape.map(\.intValue) == shape, array.dataType == .float16 else {
            throw FrameProbeError.invalid("Unexpected shape or type for \(name); expected float16 \(shape).")
        }
        return array
    }

    private func check(_ array: MLMultiArray, name: String) throws -> TensorCheck {
        // Respect output strides, including constants/broadcast tensors. Inspect only the
        // final pass, outside all measured prediction intervals.
        let shape = array.shape.map(\.intValue)
        let strides = array.strides.map(\.intValue)
        let values = array.dataPointer.assumingMemoryBound(to: Float16.self)
        var minimum = Double.infinity
        var maximum = -Double.infinity
        for flat in 0..<array.count {
            if flat % 16384 == 0 { try Task.checkCancellation() }
            var remainder = flat
            var offset = 0
            for dimension in shape.indices.reversed() {
                offset += (remainder % shape[dimension]) * strides[dimension]
                remainder /= shape[dimension]
            }
            let value = Double(values[offset])
            guard value.isFinite else { throw FrameProbeError.invalid("\(name) contains non-finite values.") }
            minimum = min(minimum, value)
            maximum = max(maximum, value)
        }
        return TensorCheck(name: name, shape: shape, minimum: minimum, maximum: maximum)
    }

    private func pixelBuffer(_ image: CGImage) throws -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        let attributes = [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary
        let status = CVPixelBufferCreate(kCFAllocatorDefault, 1024, 1024,
                                        kCVPixelFormatType_32BGRA, attributes, &buffer)
        guard status == kCVReturnSuccess, let buffer else {
            throw FrameProbeError.invalid("Could not allocate the model input image (\(status)).")
        }
        let input = CIImage(cgImage: image).transformed(by: CGAffineTransform(
            scaleX: 1024 / CGFloat(image.width), y: 1024 / CGFloat(image.height)))
        context.render(input, to: buffer, bounds: CGRect(x: 0, y: 0, width: 1024, height: 1024),
                       colorSpace: CGColorSpace(name: CGColorSpace.sRGB))
        return buffer
    }

    private func maskPNG(_ mask: MLMultiArray) throws -> (Data, Double) {
        var bytes = [UInt8](repeating: 0, count: 256 * 256 * 4)
        var foreground = 0
        for y in 0..<256 {
            for x in 0..<256 where mask[[0, 0, NSNumber(value: y), NSNumber(value: x)]].doubleValue > 0 {
                let offset = (y * 256 + x) * 4
                bytes[offset] = 90
                bytes[offset + 1] = 160
                bytes[offset + 2] = 255
                bytes[offset + 3] = 255
                foreground += 1
            }
        }
        guard let provider = CGDataProvider(data: Data(bytes) as CFData),
              let image = CGImage(width: 256, height: 256, bitsPerComponent: 8, bitsPerPixel: 32,
                                  bytesPerRow: 256 * 4, space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                                  provider: provider, decode: nil, shouldInterpolate: false,
                                  intent: .defaultIntent) else {
            throw FrameProbeError.invalid("Could not render the mask.")
        }
        return (try png(image), Double(foreground) / (256 * 256))
    }

    private func png(_ image: CGImage) throws -> Data {
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(data as CFMutableData, UTType.png.identifier as CFString, 1, nil) else {
            throw FrameProbeError.invalid("Could not create the preview image.")
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw FrameProbeError.invalid("Could not encode the preview image.")
        }
        return data as Data
    }

    private func now() -> Double { ProcessInfo.processInfo.systemUptime }
    private func milliseconds(since start: Double) -> Double { (now() - start) * 1000 }
    private func thermalState() -> String {
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
struct FirstFrameView: View {
    @State private var selection: PhotosPickerItem?
    @State private var mode: ProbeMode = .neuralEngine
    @State private var preview: UIImage?
    @State private var overlay: UIImage?
    @State private var point: CGPoint?
    @State private var running = false
    @State private var status = "Choose a video, then tap the subject in its first frame."
    @State private var reportURL: URL?
    @State private var showMask = true
    @State private var job: Task<Void, Never>?
    @State private var runner = FirstFrameRunner()

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                PhotosPicker("Choose video", selection: $selection, matching: .videos)
                    .buttonStyle(.bordered).disabled(running)
                Picker("Compute devices", selection: $mode) {
                    ForEach(ProbeMode.allCases) { Text($0.rawValue).tag($0) }
                }
                .disabled(running)
                if let preview {
                    GeometryReader { geometry in
                        ZStack(alignment: .topLeading) {
                            Image(uiImage: preview).resizable()
                            if showMask, let overlay {
                                Image(uiImage: overlay).resizable().interpolation(.none).opacity(0.45)
                            }
                            if let point {
                                Circle().stroke(.white, lineWidth: 2)
                                    .background(Circle().fill(.pink)).frame(width: 12, height: 12)
                                    .position(x: point.x * geometry.size.width, y: point.y * geometry.size.height)
                            }
                        }
                        .frame(width: geometry.size.width, height: geometry.size.height)
                        .contentShape(Rectangle())
                        .onTapGesture { location in
                            guard !running, geometry.size.width > 0, geometry.size.height > 0 else { return }
                            point = CGPoint(x: min(max(location.x / geometry.size.width, 0), 1),
                                            y: min(max(location.y / geometry.size.height, 0), 1))
                            overlay = nil
                            reportURL = nil
                            status = "Subject selected. Tap Predict mask."
                        }
                        .accessibilityLabel("Video frame. Tap a subject to select it.")
                    }
                    .aspectRatio(preview.size.width / preview.size.height, contentMode: .fit)
                    Button("Use center point") {
                        point = CGPoint(x: 0.5, y: 0.5)
                        overlay = nil
                        reportURL = nil
                        status = "Center selected. Tap Predict mask."
                    }.disabled(running)
                    Button(running ? "Working…" : "Predict mask", action: predict)
                        .buttonStyle(.borderedProminent).disabled(running || point == nil)
                }
                if running { ProgressView() }
                Text(status).textSelection(.enabled).accessibilityAddTraits(.updatesFrequently)
                if overlay != nil { Toggle("Show mask", isOn: $showMask) }
                if let reportURL { ShareLink("Share prediction report", item: reportURL) }
                Text("This tests subject selection and initial memory on one frame. It does not play or track the video yet.")
                    .font(.footnote).foregroundStyle(.secondary)
            }.padding()
        }
        .navigationTitle("First-frame prediction")
        .onChange(of: selection) { _, item in importVideo(item) }
        .onChange(of: mode) { _, _ in
            overlay = nil
            reportURL = nil
            if point != nil { status = "Tap Predict mask to test this mode." }
        }
        .onDisappear {
            job?.cancel()
            job = nil
            Task { await runner.reset() }
        }
    }

    private func importVideo(_ item: PhotosPickerItem?) {
        guard let item, !running else { return }
        running = true
        preview = nil
        overlay = nil
        point = nil
        reportURL = nil
        status = "Opening video…"
        job = Task {
            defer { running = false }
            do {
                guard let movie = try await item.loadTransferable(type: ProbeMovie.self) else {
                    throw FrameProbeError.invalid("The selected video could not be imported.")
                }
                defer { try? FileManager.default.removeItem(at: movie.url.deletingLastPathComponent()) }
                try Task.checkCancellation()
                let info = try await runner.open(url: movie.url)
                try Task.checkCancellation()
                guard let image = UIImage(data: info.png) else { throw FrameProbeError.invalid("Invalid preview.") }
                preview = image
                status = "Tap your subject, then Predict mask."
            } catch is CancellationError {
                // Navigation owns cancellation; do not publish stale state.
            } catch { status = error.localizedDescription }
        }
    }

    private func predict() {
        #if targetEnvironment(simulator)
        status = "Run this probe on your physical iPhone."
        return
        #else
        guard !running, let point, let models = Bundle.main.resourceURL?.appendingPathComponent("Models") else { return }
        let selectedMode = mode
        running = true
        overlay = nil
        reportURL = nil
        job = Task {
            defer { running = false }
            do {
                let result = try await runner.run(point: point, mode: selectedMode, modelsDirectory: models,
                                                  hardware: probeHardwareIdentifier()) { message in
                    await MainActor.run { status = message }
                }
                try Task.checkCancellation()
                overlay = result.overlayPNG.flatMap { UIImage(data: $0) }
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                encoder.dateEncodingStrategy = .iso8601
                let documents = try FileManager.default.url(for: .documentDirectory, in: .userDomainMask,
                                                            appropriateFor: nil, create: true)
                // Keep one report per mode; repeated taps do not accumulate diagnostic files.
                let suffix = selectedMode == .neuralEngine ? "cpu-neural-engine" : selectedMode == .gpu ? "cpu-gpu" : "automatic"
                let url = documents.appendingPathComponent("first-frame-\(suffix).json")
                try encoder.encode(result.report).write(to: url, options: .atomic)
                reportURL = url
                if result.report.succeeded {
                    let fraction = (result.report.foregroundFraction ?? 0) * 100
                    status = String(format: "Predictions completed. Mask covers %.1f%% of the frame. Share the report and check whether the overlay follows your selected subject.", fraction)
                    if fraction == 0 { status += " The mask is empty; try another point." }
                } else {
                    status = "Prediction failed: \(result.report.error ?? "Unknown error"). Share the report."
                }
            } catch is CancellationError {
            } catch { status = error.localizedDescription }
        }
        #endif
    }
}

private func probeHardwareIdentifier() -> String {
    var size: size_t = 0
    guard sysctlbyname("hw.machine", nil, &size, nil, 0) == 0, size > 0 else { return "Unknown" }
    var bytes = [CChar](repeating: 0, count: size)
    guard sysctlbyname("hw.machine", &bytes, &size, nil, 0) == 0 else { return "Unknown" }
    return String(cString: bytes)
}
