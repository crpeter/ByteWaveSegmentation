import SwiftUI
import PhotosUI
import AVFoundation
import CoreImage
import CoreML
import CryptoKit
import ImageIO
import UniformTypeIdentifiers
import Darwin

private struct VideoTimingTotal: Encodable {
    var count = 0
    var totalMilliseconds = 0.0
    var maximumMilliseconds = 0.0
    var meanMilliseconds = 0.0
    mutating func add(_ value: Double) {
        count += 1
        totalMilliseconds += value
        maximumMilliseconds = max(maximumMilliseconds, value)
        meanMilliseconds = totalMilliseconds / Double(count)
    }
}

private struct VideoFrameRecord: Encodable {
    let requestIndex: Int
    let segment: Int
    let ptsValue: Int64
    let ptsTimescale: Int32
    let initialized: Bool
    let foregroundFraction: Double
    let objectScore: Float
    let predictedIoU: Float
    let modelMilliseconds: [String: Double]
    let decodeAndPreparationMilliseconds: Double
    let predictionAndStateMilliseconds: Double
    let maskRenderingMilliseconds: Double
    let requestMilliseconds: Double
    let state: OwnedStateSummary
}

private struct VideoTrackingReport: Encodable {
    let schema = "bytewave.requested-video-tracking.v1"
    let createdAt = Date()
    let clipID = UUID()
    let hardware: String
    let operatingSystem = ProcessInfo.processInfo.operatingSystemVersionString
    let isIOSAppOnMac = ProcessInfo.processInfo.isiOSAppOnMac
    let modelVariant = "memoryfp16"
    let contract = OwnedTemporalContract.id
    let precisionPolicy = OwnedTemporalContract.memoryFP16Precision
    let requestedComputeUnits = "CPU_AND_GPU"
    #if DEBUG
    let swiftDebugCompilation = true
    #else
    let swiftDebugCompilation = false
    #endif
    let scope = "On-demand sequential decoded frames from a user-selected video. One positive point per reset; no ground-truth parity or automatic quality pass. All model outputs retain shape/finiteness checks. No audio, frame dropping, pre-scan or full-video mask cache. Run tracking advances as requests finish, not on a real-time playback clock."
    let timingScope = "Request time includes synchronous decode, orientation/resize, PNG preview, prediction/state, mask PNG and small pre-prediction checkpoint writes. Excludes model loading, UI presentation, report saving and user pauses. Initialization uses the already prepared displayed frame. Warm totals exclude the first prediction after every reset. These are processing diagnostics, not sustained playback FPS."
    let imagePreparation = "Decoded BGRA; track presentation transform converted to Core Image coordinates; sRGB, stretched to 1024x1024. Preview keeps display aspect. Point is normalized top-left. Graph owns image scaling/normalization."
    let recentFrameLimit = 120
    var fixtureSHA256: String?
    var diagnosticPropagator: OwnedDiagnosticPropagator?
    var sourceModelsManifestSHA256: String?
    var sourceReportSHA256: String?
    var modelFiles: [String: String] = [:]
    var compileAndLoadMilliseconds: [String: Double] = [:]
    var videoDurationSeconds = 0.0
    var sourceFileBytes: Int64 = 0
    var trackTransform: [Double] = []
    var lastDecodedSize: [Int] = []
    var lastDisplaySize: [Double] = []
    var framesPredicted = 0
    var segmentsStarted = 0
    var seeks = 0
    var lastResetReason = "new video"
    var lastPointNormalizedTopLeft: [Double]?
    var lastState: OwnedStateSummary?
    var emptyMasks = 0
    var recentFrames: [VideoFrameRecord] = []
    var warmModelTotals: [String: VideoTimingTotal] = [:]
    var warmRequestTotals = VideoTimingTotal()
    var warmPredictionAndStateTotals = VideoTimingTotal()
    var reachedEnd = false
    var lastError: String?
    var lastAction = "Opening"
    var thermalStateAtStart: String
    var thermalStateAtSnapshot: String?
}

struct OwnedVideoDisplay: Sendable {
    let previewPNG: Data
    let overlayPNG: Data?
    let seconds: Double
    let durationSeconds: Double
    let message: String
}

// Owns decoding and inference on one actor. Each next() finishes before another
// sample is requested. Only the current prepared frame and bounded model state
// survive a request; Core ML output tensors are released after rendering.
actor OwnedVideoRunner {
    private struct PreparedFrame {
        let buffer: CVPixelBuffer
        let preview: Data
        let time: CMTime
        let preparationMS: Double
    }
    private let context = CIContext(options: [.cacheIntermediates: false])
    private var generation: UInt64 = 0
    private var importedURL: URL?
    private var asset: AVURLAsset?
    private var track: AVAssetTrack?
    private var duration = CMTime.zero
    private var transform = CGAffineTransform.identity
    private var reader: AVAssetReader?
    private var output: AVAssetReaderTrackOutput?
    private var current: PreparedFrame?
    private var session: OwnedTemporalSession?
    private var compiledURLs: [URL] = []
    private var point: [Double]?
    private var report: VideoTrackingReport?
    private var checkpointURL: URL?

    // Takes ownership of the ProbeMovie temporary import, including on failure.
    func open(url: URL, modelsRoot: URL, hardware: String,
              progress: @Sendable (String) async -> Void) async throws -> OwnedVideoDisplay {
        close()
        let ticket = generation
        importedURL = url
        report = VideoTrackingReport(hardware: hardware, thermalStateAtStart: thermal())
        do {
            checkpointURL = try documents().appendingPathComponent("video-tracking-checkpoint.json")
            try checkpoint("Opening video")
            let source = AVURLAsset(url: url)
            guard let videoTrack = try await source.loadTracks(withMediaType: .video).first else {
                throw OwnedTemporalError.invalid("This file has no video track.")
            }
            let length = try await source.load(.duration)
            let preferred = try await videoTrack.load(.preferredTransform)
            try check(ticket)
            guard length.isNumeric, length.seconds.isFinite, length.seconds > 0,
                  [preferred.a, preferred.b, preferred.c, preferred.d, preferred.tx, preferred.ty].allSatisfy(\.isFinite) else {
                throw OwnedTemporalError.invalid("Invalid video duration or orientation.")
            }
            asset = source
            track = videoTrack
            duration = length
            transform = preferred
            report?.videoDurationSeconds = length.seconds
            report?.trackTransform = [preferred.a, preferred.b, preferred.c, preferred.d, preferred.tx, preferred.ty].map { Double($0) }
            report?.sourceFileBytes = (try FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?.int64Value ?? 0
            try startReader(at: .zero)
            guard let first = try readFrame() else { throw OwnedTemporalError.invalid("The video contains no decodable frames.") }
            current = first

            let data = try Data(contentsOf: modelsRoot.appendingPathComponent("fixture.json"))
            let fixture = try JSONDecoder().decode(OwnedFixture.self, from: data)
            try fixture.diagnosticPropagator?.validate()
            guard fixture.schema == "bytewave.temporal-device-fixture.v1",
                  fixture.contract == OwnedTemporalContract.id,
                  fixture.graphRevision == OwnedTemporalContract.graphRevision,
                  fixture.upstream == OwnedTemporalContract.upstream,
                  fixture.checkpointSHA256 == OwnedTemporalContract.checkpoint,
                  fixture.precisionPolicy == OwnedTemporalContract.memoryFP16Precision,
                  fixture.diagnosticPropagator?.variant == "memoryfp16" else {
                throw OwnedTemporalError.invalid("Prepare the validated Memory FP16 candidate before tracking video.")
            }
            report?.fixtureSHA256 = digest(data)
            report?.diagnosticPropagator = fixture.diagnosticPropagator
            report?.sourceModelsManifestSHA256 = fixture.sourceModelsManifestSHA256
            report?.sourceReportSHA256 = fixture.sourceReportSHA256
            report?.modelFiles = fixture.modelFiles
            // Check the exact package file set, not merely a nonempty hash list.
            var actualPaths = Set<String>()
            for component in OwnedTemporalContract.components {
                let prefix = "models/BWTemporal\(component).mlpackage/"
                let package = modelsRoot.appendingPathComponent(String(prefix.dropLast()))
                guard let files = FileManager.default.enumerator(at: package, includingPropertiesForKeys: [.isRegularFileKey],
                                                                 options: []) else {
                    throw OwnedTemporalError.invalid("Missing model package: \(component).")
                }
                for case let file as URL in files {
                    if try file.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true {
                        let relative = prefix + String(file.path.dropFirst(package.path.count + 1))
                        guard let expected = fixture.modelFiles[relative], digest(try Data(contentsOf: file, options: .mappedIfSafe)) == expected else {
                            throw OwnedTemporalError.invalid("Model file verification failed: \(relative).")
                        }
                        actualPaths.insert(relative)
                    }
                    try check(ticket)
                }
            }
            guard !actualPaths.isEmpty, actualPaths == Set(fixture.modelFiles.keys) else {
                throw OwnedTemporalError.invalid("Model package file set does not match its manifest.")
            }
            var models: [String: MLModel] = [:]
            // Local URLs stay owned by this open operation across its awaits.
            var temporaryCompiled: [URL] = []
            var adopted = false
            defer { if !adopted { for file in temporaryCompiled { try? FileManager.default.removeItem(at: file) } } }
            for component in OwnedTemporalContract.components {
                try check(ticket)
                try checkpoint("Loading \(component)")
                await progress("Loading \(component)…")
                try check(ticket)
                let start = now()
                let compiled = try await MLModel.compileModel(at: modelsRoot.appendingPathComponent("models/BWTemporal\(component).mlpackage"))
                temporaryCompiled.append(compiled)
                try check(ticket)
                let configuration = MLModelConfiguration()
                configuration.computeUnits = .cpuAndGPU
                let model = try await MLModel.load(contentsOf: compiled, configuration: configuration)
                try check(ticket)
                try OwnedTemporalContract.validate(model, component: component, propagatorVariant: "memoryfp16")
                models[component] = model
                report?.compileAndLoadMilliseconds[component] = elapsed(start)
            }
            session = OwnedTemporalSession(models: models)
            compiledURLs = temporaryCompiled
            adopted = true
            report?.lastAction = "Ready for selection"
            return display(first, overlay: nil, message: "Tap the subject, then Track selected subject.")
        } catch {
            if ticket == generation {
                report?.lastError = error.localizedDescription
                report?.lastAction = "Open failed"
                releaseResources()
            }
            throw error
        }
    }

    func initialize(point selected: [Double]) throws -> OwnedVideoDisplay {
        guard selected.count == 2, selected.allSatisfy({ $0.isFinite && (0...1).contains($0) }),
              current != nil, let session else { throw OwnedTemporalError.invalid("Choose a frame and tap the subject.") }
        session.reset()
        point = selected
        report?.segmentsStarted += 1
        report?.lastResetReason = "subject selection on current frame"
        report?.lastPointNormalizedTopLeft = selected
        return try predictCurrent(initialized: true, requestStart: now(), preparationMS: 0)
    }

    func next() throws -> OwnedVideoDisplay? {
        guard point != nil else { throw OwnedTemporalError.invalid("Select a subject before advancing.") }
        let start = now()
        do {
            try Task.checkCancellation()
            guard let frame = try readFrame() else {
                report?.reachedEnd = true
                report?.lastAction = "End of video"
                return nil
            }
            if let previous = current, CMTimeCompare(frame.time, previous.time) <= 0 {
                throw OwnedTemporalError.invalid("Decoded timestamps did not advance. Seek and reselect the subject.")
            }
            current = frame
            return try predictCurrent(initialized: false, requestStart: start, preparationMS: frame.preparationMS)
        } catch {
            invalidatePrediction(error)
            throw error
        }
    }

    func seek(seconds: Double) throws -> OwnedVideoDisplay {
        guard seconds.isFinite, seconds >= 0, seconds < duration.seconds else {
            throw OwnedTemporalError.invalid("Choose a time before the end of the video.")
        }
        session?.reset()
        point = nil
        current = nil
        report?.seeks += 1
        report?.lastResetReason = "seek; new selection required"
        report?.lastState = nil
        report?.reachedEnd = false
        do {
            let time = CMTime(seconds: seconds, preferredTimescale: 60000)
            try startReader(at: time)
            guard let frame = try readFrame(notBefore: time) else {
                throw OwnedTemporalError.invalid("No frame at this time. Seek earlier.")
            }
            current = frame
            report?.lastAction = "Seek completed; awaiting selection"
            return display(frame, overlay: nil, message: "Tracking reset. Tap the subject in this frame.")
        } catch {
            invalidatePrediction(error)
            throw error
        }
    }

    func saveReport() throws -> URL {
        guard var snapshot = report else { throw OwnedTemporalError.invalid("No video report yet.") }
        snapshot.thermalStateAtSnapshot = thermal()
        let url = try documents().appendingPathComponent("video-tracking-last-run.json")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(snapshot).write(to: url, options: .atomic)
        return url
    }

    func close() {
        generation &+= 1
        if report != nil { _ = try? saveReport() }
        releaseResources()
        report = nil
    }

    private func releaseResources() {
        reader?.cancelReading()
        reader = nil
        output = nil
        current = nil
        asset = nil
        track = nil
        session?.reset()
        session = nil
        point = nil
        for url in compiledURLs { try? FileManager.default.removeItem(at: url) }
        compiledURLs.removeAll()
        if let importedURL { try? FileManager.default.removeItem(at: importedURL.deletingLastPathComponent()) }
        importedURL = nil
        context.clearCaches()
    }

    private func startReader(at time: CMTime) throws {
        guard let asset, let track else { throw OwnedTemporalError.invalid("No video is open.") }
        reader?.cancelReading()
        let nextReader = try AVAssetReader(asset: asset)
        let nextOutput = AVAssetReaderTrackOutput(track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        nextOutput.alwaysCopiesSampleData = false
        guard nextReader.canAdd(nextOutput) else { throw OwnedTemporalError.invalid("Cannot decode this video.") }
        nextReader.add(nextOutput)
        nextReader.timeRange = CMTimeRange(start: time, duration: CMTimeSubtract(duration, time))
        guard nextReader.startReading() else { throw nextReader.error ?? OwnedTemporalError.invalid("Video decoding could not start.") }
        reader = nextReader
        output = nextOutput
    }

    private func readFrame(notBefore: CMTime? = nil) throws -> PreparedFrame? {
        guard let reader, let output else { throw OwnedTemporalError.invalid("The decoder is not ready.") }
        let start = now()
        while true {
            try Task.checkCancellation()
            var skippedPreroll = false
            // Temporary native buffers and rendering intermediates leave this pool.
            let decoded: PreparedFrame? = try autoreleasepool {
                guard let sample = output.copyNextSampleBuffer() else { return nil }
                let time = CMSampleBufferGetPresentationTimeStamp(sample)
                guard time.isNumeric, time.epoch == 0, time.timescale > 0 else {
                    throw OwnedTemporalError.invalid("Invalid decoded presentation timestamp.")
                }
                if let notBefore, CMTimeCompare(time, notBefore) < 0 {
                    skippedPreroll = true
                    return nil
                }
                guard let buffer = CMSampleBufferGetImageBuffer(sample) else { throw OwnedTemporalError.invalid("Missing decoded pixels.") }
                // Track matrices use presentation coordinates. Flip into that
                // coordinate system, apply rotation/mirroring, then back to CI.
                let source = CIImage(cvPixelBuffer: buffer)
                let flipped = source.transformed(by: CGAffineTransform(a: 1, b: 0, c: 0, d: -1, tx: 0, ty: source.extent.height))
                let presented = flipped.transformed(by: transform)
                let bounds = presented.extent
                guard !bounds.isInfinite, !bounds.isNull, bounds.width > 0, bounds.height > 0 else {
                    throw OwnedTemporalError.invalid("Invalid display-oriented frame size.")
                }
                report?.lastDecodedSize = [CVPixelBufferGetWidth(buffer), CVPixelBufferGetHeight(buffer)]
                report?.lastDisplaySize = [Double(bounds.width), Double(bounds.height)]
                let normalized = presented.transformed(by: CGAffineTransform(translationX: -bounds.minX, y: -bounds.minY))
                let oriented = normalized.transformed(by: CGAffineTransform(a: 1, b: 0, c: 0, d: -1, tx: 0, ty: bounds.height))
                var input: CVPixelBuffer?
                let code = CVPixelBufferCreate(kCFAllocatorDefault, 1024, 1024, kCVPixelFormatType_32BGRA,
                                              [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &input)
                guard code == kCVReturnSuccess, let input else { throw OwnedTemporalError.invalid("Could not allocate the model image.") }
                let color = CGColorSpace(name: CGColorSpace.sRGB)!
                let square = oriented.transformed(by: CGAffineTransform(scaleX: 1024 / bounds.width, y: 1024 / bounds.height))
                context.render(square, to: input, bounds: CGRect(x: 0, y: 0, width: 1024, height: 1024), colorSpace: color)
                let scale = min(1, 1024 / max(bounds.width, bounds.height))
                let preview = oriented.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
                guard let image = context.createCGImage(preview, from: preview.extent, format: .RGBA8, colorSpace: color) else {
                    throw OwnedTemporalError.invalid("Could not prepare the video preview.")
                }
                return PreparedFrame(buffer: input, preview: try png(image), time: time, preparationMS: elapsed(start))
            }
            if let decoded { return decoded }
            switch reader.status {
            case .completed: return nil
            case .failed: throw reader.error ?? OwnedTemporalError.invalid("Video decoding failed.")
            case .cancelled: throw CancellationError()
            case .reading:
                // Only a preroll sample skipped by notBefore can reach here
                // while reading. A nil sample at EOF normally marks completed.
                if skippedPreroll { continue }
                throw OwnedTemporalError.invalid("Decoder returned no frame before completion.")
            default: throw OwnedTemporalError.invalid("Unexpected video decoder state.")
            }
        }
    }

    private func predictCurrent(initialized: Bool, requestStart: Double, preparationMS: Double) throws -> OwnedVideoDisplay {
        guard let current, let session, let point else { throw OwnedTemporalError.invalid("Select a subject first.") }
        do {
            return try autoreleasepool {
                let start = now()
                let prediction = try session.predict(image: current.buffer, at: current.time, initialPoint: point) { component in
                    try checkpoint("Before prediction: \(component)", time: current.time)
                }
                let predictionMS = elapsed(start)
                guard let mask = prediction.tensors["low_res_mask"], let score = prediction.tensors["object_score"],
                      let iou = prediction.tensors["best_iou"] else { throw OwnedTemporalError.invalid("Incomplete prediction.") }
                let maskStart = now()
                let overlay = try maskPNG(mask.values)
                let maskMS = elapsed(maskStart)
                try Task.checkCancellation()
                let fraction = Double(mask.values.lazy.filter { $0 > 0 }.count) / Double(mask.values.count)
                let row = VideoFrameRecord(requestIndex: report?.framesPredicted ?? 0, segment: report?.segmentsStarted ?? 0,
                    ptsValue: current.time.value, ptsTimescale: current.time.timescale, initialized: initialized,
                    foregroundFraction: fraction, objectScore: score.values[0], predictedIoU: iou.values[0],
                    modelMilliseconds: prediction.modelMilliseconds, decodeAndPreparationMilliseconds: preparationMS,
                    predictionAndStateMilliseconds: predictionMS, maskRenderingMilliseconds: maskMS,
                    requestMilliseconds: elapsed(requestStart), state: prediction.state)
                report?.framesPredicted += 1
                report?.lastState = prediction.state
                if fraction == 0 { report?.emptyMasks += 1 }
                report?.recentFrames.append(row)
                if (report?.recentFrames.count ?? 0) > 120 { report?.recentFrames.removeFirst() }
                if !initialized {
                    for (name, ms) in prediction.modelMilliseconds { report?.warmModelTotals[name, default: VideoTimingTotal()].add(ms) }
                    report?.warmRequestTotals.add(row.requestMilliseconds)
                    report?.warmPredictionAndStateTotals.add(predictionMS)
                }
                report?.lastAction = "Predicted frame at \(current.time.seconds) seconds"
                let message = String(format: "%.3f s · mask %.1f%% · prediction %.1f ms · state %d/7 + %d/16",
                                     current.time.seconds, fraction * 100, predictionMS,
                                     prediction.state.spatialEntries, prediction.state.pointerEntries)
                return display(current, overlay: overlay, message: message)
            }
        } catch {
            invalidatePrediction(error)
            throw error
        }
    }

    private func invalidatePrediction(_ error: Error) {
        session?.reset()
        point = nil
        report?.lastState = nil
        report?.lastResetReason = "error or cancellation"
        report?.lastError = error.localizedDescription
        report?.lastAction = "Stopped; new selection required"
    }

    private func display(_ frame: PreparedFrame, overlay: Data?, message: String) -> OwnedVideoDisplay {
        OwnedVideoDisplay(previewPNG: frame.preview, overlayPNG: overlay, seconds: frame.time.seconds,
                          durationSeconds: duration.seconds, message: message)
    }
    private func check(_ ticket: UInt64) throws {
        try Task.checkCancellation()
        guard ticket == generation else { throw CancellationError() }
    }
    private func checkpoint(_ stage: String, time: CMTime? = nil) throws {
        guard let checkpointURL else { return }
        let data: [String: Any] = ["schema": "bytewave.video-checkpoint.v1", "stage": stage,
            "modelVariant": "memoryfp16", "fixtureSHA256": report?.fixtureSHA256 ?? "",
            "framesPredicted": report?.framesPredicted ?? 0,
            "ptsValue": time?.value ?? 0, "ptsTimescale": time?.timescale ?? 1]
        try JSONSerialization.data(withJSONObject: data, options: [.sortedKeys]).write(to: checkpointURL, options: .atomic)
    }
    private func documents() throws -> URL {
        try FileManager.default.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
    }
    private func digest(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
    private func now() -> Double { ProcessInfo.processInfo.systemUptime }
    private func elapsed(_ start: Double) -> Double { (now() - start) * 1000 }
    private func thermal() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }
    private func png(_ image: CGImage) throws -> Data {
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(data as CFMutableData, UTType.png.identifier as CFString, 1, nil) else {
            throw OwnedTemporalError.invalid("Could not create PNG.")
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { throw OwnedTemporalError.invalid("Could not encode PNG.") }
        return data as Data
    }
    private func maskPNG(_ values: [Float]) throws -> Data {
        var bytes = [UInt8](repeating: 0, count: 256 * 256 * 4)
        for i in values.indices where values[i] > 0 {
            bytes[i * 4] = 90; bytes[i * 4 + 1] = 160; bytes[i * 4 + 2] = 255; bytes[i * 4 + 3] = 255
        }
        guard let provider = CGDataProvider(data: Data(bytes) as CFData),
              let image = CGImage(width: 256, height: 256, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: 1024,
                  space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                  provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent) else {
            throw OwnedTemporalError.invalid("Could not draw mask.")
        }
        return try png(image)
    }
}

@MainActor
struct OwnedVideoTrackingView: View {
    @Environment(\.scenePhase) private var scenePhase
    @State private var selection: PhotosPickerItem?
    @State private var runner = OwnedVideoRunner()
    @State private var job: Task<Void, Never>?
    @State private var lifecycle = UUID()
    @State private var busy = false
    @State private var playing = false
    @State private var pauseRequested = false
    @State private var preview: UIImage?
    @State private var overlay: UIImage?
    @State private var point: CGPoint?
    @State private var tracking = false
    @State private var ended = false
    @State private var showMask = true
    @State private var seconds = 0.0
    @State private var duration = 0.0
    @State private var seekSeconds = 0.0
    @State private var status = "Choose a video, then tap the subject."
    @State private var reportURL: URL?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                PhotosPicker("Choose video", selection: $selection, matching: .videos).disabled(busy)
                Text("Memory FP16 · CPU + GPU").font(.subheadline).foregroundStyle(.secondary)
                if let preview {
                    GeometryReader { geometry in
                        ZStack(alignment: .topLeading) {
                            Image(uiImage: preview).resizable()
                            if showMask, let overlay { Image(uiImage: overlay).resizable().interpolation(.none).opacity(0.45) }
                            if !tracking, let point {
                                Circle().fill(.pink).overlay(Circle().stroke(.white, lineWidth: 2))
                                    .frame(width: 12, height: 12)
                                    .position(x: point.x * geometry.size.width, y: point.y * geometry.size.height)
                            }
                        }
                        .frame(width: geometry.size.width, height: geometry.size.height)
                        .contentShape(Rectangle())
                        .onTapGesture { location in
                            guard !busy, geometry.size.width > 0, geometry.size.height > 0 else { return }
                            point = CGPoint(x: min(max(location.x / geometry.size.width, 0), 1),
                                            y: min(max(location.y / geometry.size.height, 0), 1))
                            tracking = false
                            overlay = nil
                            reportURL = nil
                            status = "Tap Track selected subject to start a fresh selection here."
                        }
                        .accessibilityLabel("Current video frame. Tap the subject to select it.")
                    }.aspectRatio(preview.size.width / preview.size.height, contentMode: .fit)
                    Button("Use center point") {
                        point = CGPoint(x: 0.5, y: 0.5); tracking = false; overlay = nil; reportURL = nil
                    }.disabled(busy)
                    Button("Track selected subject") { initialize() }
                        .buttonStyle(.borderedProminent).disabled(busy || point == nil || tracking)
                    HStack {
                        Button("Next frame") { advance(continuous: false) }.disabled(busy || !tracking || ended)
                        Button(playing ? "Pause" : "Run tracking") {
                            if playing { pauseRequested = true } else { advance(continuous: true) }
                        }.disabled(playing ? pauseRequested : (busy || !tracking || ended))
                    }.buttonStyle(.bordered)
                    Toggle("Show mask", isOn: $showMask)
                }
                if duration > 0 {
                    Text(String(format: "Frame %.3f s / %.3f s", seconds, duration)).monospacedDigit()
                    Slider(value: $seekSeconds, in: 0...max(duration - 0.001, 0.001)).disabled(busy)
                    HStack {
                        Button(String(format: "Seek to %.2f s", seekSeconds)) { seek(to: seekSeconds) }
                        Button("Restart") { seek(to: 0) }
                    }.disabled(busy)
                    Text("Seeking resets tracking. Select the subject again at the new time.").font(.footnote)
                }
                if busy { ProgressView() }
                Text(status).textSelection(.enabled)
                if let reportURL { ShareLink("Share video tracking report", item: reportURL).disabled(busy) }
                Text("Frames advance as processing finishes. Audio is off. Check that the blue mask follows your subject.")
                    .font(.footnote).foregroundStyle(.secondary)
            }.padding()
        }
        .navigationTitle("Track a video")
        .onChange(of: selection) { _, item in open(item) }
        .onChange(of: scenePhase) { _, phase in
            if phase != .active { pauseRequested = true }
        }
        .onDisappear {
            lifecycle = UUID()
            job?.cancel()
            busy = false
            playing = false
            tracking = false
            preview = nil
            overlay = nil
            point = nil
            duration = 0
            selection = nil
            status = "Choose a video, then tap the subject."
            Task { await runner.close() }
        }
    }

    private func open(_ item: PhotosPickerItem?) {
        #if targetEnvironment(simulator)
        status = "Run video tracking on a physical iPhone."
        #else
        guard let item, !busy, let root = Bundle.main.resourceURL?.appendingPathComponent("DeviceValidationData/memoryfp16") else { return }
        let ticket = lifecycle
        busy = true; preview = nil; overlay = nil; point = nil; tracking = false; ended = false
        duration = 0; reportURL = nil; status = "Opening video…"
        job = Task {
            defer { if ticket == lifecycle { busy = false; selection = nil } }
            do {
                // Close the old decoder/import before Photos copies the next clip.
                await runner.close()
                guard let movie = try await item.loadTransferable(type: ProbeMovie.self) else {
                    throw OwnedTemporalError.invalid("The video could not be imported.")
                }
                var handedOff = false
                defer { if !handedOff { try? FileManager.default.removeItem(at: movie.url.deletingLastPathComponent()) } }
                try Task.checkCancellation()
                guard ticket == lifecycle else { return }
                handedOff = true
                let result = try await runner.open(url: movie.url, modelsRoot: root, hardware: hardware()) { message in
                    await MainActor.run { if ticket == lifecycle { status = message } }
                }
                try Task.checkCancellation()
                guard ticket == lifecycle else { return }
                apply(result)
            } catch {
                guard ticket == lifecycle else { return }
                status = error.localizedDescription
            }
            guard ticket == lifecycle, !Task.isCancelled else { return }
            let saved = try? await runner.saveReport()
            if ticket == lifecycle, !Task.isCancelled { reportURL = saved }
        }
        #endif
    }

    private func initialize() {
        guard !busy, let point else { return }
        perform {
            let result = try await runner.initialize(point: [Double(point.x), Double(point.y)])
            try Task.checkCancellation()
            apply(result)
            tracking = true
        }
    }

    private func seek(to value: Double) {
        guard !busy else { return }
        preview = nil; overlay = nil; point = nil; tracking = false; ended = false
        perform { apply(try await runner.seek(seconds: value)) }
    }

    private func advance(continuous: Bool) {
        guard !busy, tracking, !ended else { return }
        playing = continuous
        pauseRequested = false
        perform {
            repeat {
                try Task.checkCancellation()
                let next = try await runner.next()
                try Task.checkCancellation()
                guard let result = next else {
                    ended = true
                    status = "End of video. Share the report and review whether tracking stayed on your subject."
                    break
                }
                try Task.checkCancellation()
                apply(result)
                // A single request is in flight. Pause finishes that request so
                // displayed pixels, mask and committed state always agree.
                await Task.yield()
            } while continuous && !pauseRequested
        }
    }

    private func perform(_ action: @escaping @MainActor () async throws -> Void) {
        let ticket = lifecycle
        busy = true
        reportURL = nil
        job = Task {
            defer { if ticket == lifecycle { busy = false; playing = false } }
            do {
                try Task.checkCancellation()
                try await action()
            } catch {
                guard ticket == lifecycle else { return }
                // The actor may already hold the next decoded frame. Hide the
                // previous image rather than permit selection on stale pixels.
                preview = nil; overlay = nil; point = nil; tracking = false
                status = "\(error.localizedDescription) Use Restart or seek to recover."
            }
            guard ticket == lifecycle, !Task.isCancelled else { return }
            do {
                let saved = try await runner.saveReport()
                if ticket == lifecycle, !Task.isCancelled { reportURL = saved }
            } catch {
                if ticket == lifecycle, !Task.isCancelled { status += " Report could not be saved: \(error.localizedDescription)" }
            }
        }
    }

    private func apply(_ result: OwnedVideoDisplay) {
        guard !Task.isCancelled else { return }
        preview = UIImage(data: result.previewPNG)
        overlay = result.overlayPNG.flatMap { UIImage(data: $0) }
        seconds = result.seconds
        seekSeconds = min(result.seconds, max(result.durationSeconds - 0.001, 0))
        duration = result.durationSeconds
        status = result.message
    }
    private func hardware() -> String {
        var size: size_t = 0
        guard sysctlbyname("hw.machine", nil, &size, nil, 0) == 0, size > 0 else { return "Unknown" }
        var bytes = [CChar](repeating: 0, count: size)
        guard sysctlbyname("hw.machine", &bytes, &size, nil, 0) == 0 else { return "Unknown" }
        return String(cString: bytes)
    }
}
