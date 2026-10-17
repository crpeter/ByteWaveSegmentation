import SwiftUI
import PhotosUI
import AVFoundation
import CoreImage
import CoreML
import CryptoKit
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
    let subjectID: UUID
    let segment: Int
    let ptsValue: Int64
    let ptsTimescale: Int32
    let initialized: Bool
    let foregroundFraction: Double
    let objectScore: Float
    let predictedIoU: Float
    let modelMilliseconds: [String: Double]
    let sessionStageMilliseconds: [String: Double]
    let decodeAndPreparationMilliseconds: Double
    let preparationStageMilliseconds: [String: Double]
    let predictionAndStateMilliseconds: Double
    let maskRenderingMilliseconds: Double
    let requestMilliseconds: Double
    let state: OwnedStateSummary
}

private struct VideoCorrectionRecord: Encodable {
    let subjectID: UUID
    let fromPTSValue: Int64
    let fromPTSTimescale: Int32
    let retainedMaskFrames: Int
    let invalidatedMaskFrames: Int
    // Metadata digest; replay separately verifies the retained bytes on disk.
    let retainedPrefixMetadataSHA256: String
    var replacementPTSValue: Int64?
    var replacementPTSTimescale: Int32?
    var replacementSegment: Int?
}

private struct VideoCachedSegmentRecord: Encodable {
    let subjectID: UUID
    let segment: Int
    var masks: Int
    let firstPTSValue: Int64
    let firstPTSTimescale: Int32
    var lastPTSValue: Int64
    var lastPTSTimescale: Int32
}

private struct VideoTrackingReport: Encodable {
    let schema = "bytewave.requested-video-tracking.v2"
    let createdAt = Date()
    let clipID = UUID()
    let hardware: String
    let operatingSystem = ProcessInfo.processInfo.operatingSystemVersionString
    let isIOSAppOnMac = ProcessInfo.processInfo.isiOSAppOnMac
    let modelVariant = "memoryfp16projection"
    let contract = OwnedTemporalContract.id
    let precisionPolicy = OwnedTemporalContract.encoderProjectionFP16Precision
    let requestedComputeUnits = "CPU_AND_GPU"
    #if DEBUG
    let swiftDebugCompilation = true
    #else
    let swiftDebugCompilation = false
    #endif
    let scope = "On-demand sequential decoded frames from a user-selected video. One logical subject can have multiple manually corrected tracking segments, each initialized with one positive point and fresh model memory. No automatic identity/loss decision or ground-truth quality pass. Shape/finiteness checks remain. No audio, frame dropping or pre-scan. Bounded subject mask cache supports replay paced by source timestamps."
    let timingScope = "Request time includes synchronous decode, orientation/resize, preview image rendering, prediction/state, mask image creation, mask packing/hash/disk write and small pre-prediction checkpoint writes. Excludes model loading, UI presentation, report saving and user pauses. Initialization uses the already prepared displayed frame. Warm totals exclude initialization after every reset. Replay is measured separately. These are processing diagnostics, not sustained playback FPS."
    let timingInstrumentation = "video-mask-cache.v1"
    let sessionTimingInstrumentation = "session-stages.v1"
    let tensorCopyImplementation = OwnedTensor.copyImplementation
    let sessionTimingScope = "Input preparation, output copy/validation, before-prediction checkpoint hooks and state pack/commit are timed separately inside predictionAndStateMilliseconds. Stages exclude model calls and do not sum to the full session time; orchestration overhead remains. Warm totals exclude initialization after every reset."
    let previewTransport = "Video preview and mask overlay use owned sRGB RGBA8 CGImages shared with UI; no PNG encoding or decoding."
    let maskPreviewImplementation = "high-res-threshold-cgimage.v1"
    let maskPreview = "Existing high_res_mask [1,1,1024,1024], threshold > 0 after model upsampling; low-quality image interpolation for display, blue overlay opacity 0.45. Presentation only: no inference, state or selection change; not hair matting."
    let foregroundFractionSource = "low_res_mask > 0; unchanged source for foregroundFraction and emptyMasks diagnostics."
    let preparationTimingScope = "Non-overlapping API wall times within decodeAndPreparationMilliseconds: sample acquisition, orientation graph setup, model buffer allocation, model input render and eager preview image creation. Preview rendering uses deferred=false before handing the image to the UI. These are API wall times, not hardware execution times. Sample acquisition includes waiting for decoded pixels, not isolated decoder execution. Small bookkeeping and autorelease cleanup are not separate stages. Initialization has an empty stage dictionary because it reuses the displayed frame. Warm preparation totals include only successful next-frame predictions; exclude open/seek previews and initialization. UI image construction, SwiftUI rendering and presentation remain outside request timing."
    let imagePreparation = "Decoded BGRA; track presentation transform converted to Core Image coordinates; sRGB, stretched to 1024x1024. Preview keeps display aspect. Point is normalized top-left. Graph owns image scaling/normalization."
    let recentFrameLimit = 120
    var fixtureSHA256: String?
    var diagnosticPropagator: OwnedDiagnosticPropagator?
    var diagnosticEncoder: OwnedDiagnosticEncoder?
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
    var subjectID: UUID?
    var subjectsStarted = 0
    var correctionsStarted = 0
    var correctionsCompleted = 0
    var pendingCorrectionFromSeconds: Double?
    var recentCorrections: [VideoCorrectionRecord] = [] // Last 120; subject IDs distinguish new tracks.
    var seeks = 0
    var lastResetReason = "new video"
    var lastPointNormalizedTopLeft: [Double]?
    var lastState: OwnedStateSummary?
    var emptyMasks = 0
    var recentFrames: [VideoFrameRecord] = []
    var warmModelTotals: [String: VideoTimingTotal] = [:]
    var warmSessionStageTotals: [String: VideoTimingTotal] = [:]
    var warmRequestTotals = VideoTimingTotal()
    var warmPredictionAndStateTotals = VideoTimingTotal()
    var warmDecodeAndPreparationTotals = VideoTimingTotal()
    var warmPreparationStageTotals: [String: VideoTimingTotal] = [:]
    var warmMaskRenderingTotals = VideoTimingTotal()
    let maskCacheImplementation = "subject-correction-bitmask-replay.v2"
    let maskCacheScope = "One logical subject; exact source PTS, lossless packed high_res_mask > 0, subject and segment ownership. Up to 256 MiB of mask payload on temporary disk. Seeking preserves masks but resets inference memory. Manual correction discards masks at/after the marked frame, retains the prefix, and initializes a fresh segment at the reselected frame. Missing masks replay as explicitly untracked, not predicted absence. New subject/video clears the cache. No persistent project cache, audio or export integration. Replay never calls models or advances inference state."
    var cachedMaskFrames = 0
    var cachedMaskBytes = 0
    var cacheSelectionSegment: Int?
    var cachedStartSeconds: Double?
    var cachedEndSeconds: Double?
    var cacheTimelineStartSeconds: Double?
    var cacheTimelineEndSeconds: Double?
    var cachedSegments: [VideoCachedSegmentRecord] = [] // Populated from the bounded cache at report save.
    var cacheLimitReached = false
    var cacheError: String?
    var cacheWriteTotals = VideoTimingTotal()
    var replayFrames = 0
    var replayRuns = 0
    var lastReplayPredictionCountBefore: Int?
    var lastReplayPredictionCountAfter: Int?
    var lastReplayStateBefore: OwnedStateSummary?
    var lastReplayStateAfter: OwnedStateSummary?
    var lastReplayFramesRead = 0
    var lastReplayMasksRead = 0
    var lastReplayUntrackedFrames = 0
    var lastReplaySubjectIDBefore: UUID?
    var lastReplaySubjectIDAfter: UUID?
    var replayError: String?
    var replayRequestTotals = VideoTimingTotal()
    let replayTimingScope = "Replay request includes video decode/orientation/preview, disk mask read/hash verification and overlay construction. Excludes presentation pacing, UI rendering and report saving. Inference totals exclude replay."
    var reachedEnd = false
    var lastError: String?
    var lastAction = "Opening"
    var thermalStateAtStart: String
    var thermalStateAtSnapshot: String?
}

struct OwnedVideoDisplay: Sendable {
    let previewImage: CGImage
    let overlayImage: CGImage?
    let seconds: Double
    let durationSeconds: Double
    let message: String
    let cachedFrames: Int
    let cacheNotice: String?
    let hasSubject: Bool
    let isTracking: Bool
    let correctionFromSeconds: Double?
    let canReplay: Bool
}

// Owns decoding and inference on one actor. Each next() finishes before another
// sample is requested. Only the current prepared frame and bounded model state
// survive a request; Core ML output tensors are released after rendering.
// Logical subject identity survives model resets and manual correction segments.
// A bounded disk cache keeps the valid prefix; gaps never masquerade as masks.
actor OwnedVideoRunner {
    private struct PreparedFrame {
        let buffer: CVPixelBuffer?
        let preview: CGImage
        let time: CMTime
        let preparationMS: Double
        let preparationStages: [String: Double]
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
    private var subjectID: UUID?
    private var correctionStart: CMTime?
    private var report: VideoTrackingReport?
    private var checkpointURL: URL?
    private struct CachedMask {
        let time: CMTime
        let sha256: String
        let subjectID: UUID
        let segment: Int
    }
    private static let maskByteCount = 1024 * 1024 / 8
    private static let maximumCachedFrames = 2048 // 256 MiB payload, bounded metadata.
    private var cacheDirectory: URL?
    private var cacheWriter: FileHandle?
    private var cachedMasks: [CachedMask] = []
    private var cacheTimelineStart: CMTime?
    private var cacheTimelineEnd: CMTime?
    private var currentOverlay: CGImage?
    private var replayReader: AVAssetReader?
    private var replayOutput: AVAssetReaderTrackOutput?
    private var replayFile: FileHandle?
    private var replayIndex = 0
    private var replayLastTime: CMTime?

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
            try fixture.diagnosticEncoder?.validate()
            guard fixture.schema == "bytewave.temporal-device-fixture.v1",
                  fixture.contract == OwnedTemporalContract.id,
                  fixture.graphRevision == OwnedTemporalContract.graphRevision,
                  fixture.upstream == OwnedTemporalContract.upstream,
                  fixture.checkpointSHA256 == OwnedTemporalContract.checkpoint,
                  fixture.precisionPolicy == OwnedTemporalContract.encoderProjectionFP16Precision,
                  fixture.diagnosticEncoder?.variant == "projection",
                  fixture.diagnosticPropagator?.variant == "memoryfp16" else {
                throw OwnedTemporalError.invalid("Prepare the validated Memory FP16 + fused encoder fixture before tracking video.")
            }
            report?.fixtureSHA256 = digest(data)
            report?.diagnosticPropagator = fixture.diagnosticPropagator
            report?.diagnosticEncoder = fixture.diagnosticEncoder
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
                try OwnedTemporalContract.validate(model, component: component,
                                                   propagatorVariant: "memoryfp16", encoderVariant: "projection")
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
              let current, let session else { throw OwnedTemporalError.invalid("Choose a frame and tap the subject.") }
        try Task.checkCancellation()
        let correcting = subjectID != nil
        if correcting {
            guard let correctionStart, CMTimeCompare(current.time, correctionStart) >= 0 else {
                throw OwnedTemporalError.invalid("Mark where tracking went wrong, then select the same subject at that frame or later.")
            }
            // Repair a failed truncate/write tail before appending; never extend
            // a short file with zeros or publish a mask over retained entries.
            try reopenCacheWriter()
        } else {
            resetMaskCache(create: true)
            guard cacheWriter != nil else { throw OwnedTemporalError.invalid("The mask cache could not be created. Try starting again.") }
            let identity = UUID()
            subjectID = identity
            report?.subjectID = identity
            report?.subjectsStarted += 1
        }
        session.reset()
        currentOverlay = nil
        point = selected
        report?.segmentsStarted += 1
        report?.lastResetReason = correcting ? "same subject; fresh memory after manual correction" : "new subject selection"
        report?.lastPointNormalizedTopLeft = selected
        report?.reachedEnd = false
        let result: OwnedVideoDisplay
        do {
            result = try predictCurrent(initialized: true, requestStart: now(), preparationMS: 0)
        } catch {
            if !correcting {
                subjectID = nil
                report?.subjectID = nil
            }
            throw error
        }
        if correcting, var snapshot = report, !snapshot.recentCorrections.isEmpty {
            let index = snapshot.recentCorrections.count - 1
            let segment = snapshot.segmentsStarted
            snapshot.recentCorrections[index].replacementPTSValue = current.time.value
            snapshot.recentCorrections[index].replacementPTSTimescale = current.time.timescale
            snapshot.recentCorrections[index].replacementSegment = segment
            snapshot.correctionsCompleted += 1
            snapshot.pendingCorrectionFromSeconds = nil
            report = snapshot
        }
        correctionStart = nil
        return display(current, overlay: currentOverlay, message: result.message)
    }

    func correctFromCurrentFrame() throws -> OwnedVideoDisplay {
        guard let current, let subjectID, correctionStart == nil else {
            throw OwnedTemporalError.invalid("Choose the first incorrect frame before starting a correction.")
        }
        try Task.checkCancellation()
        stopReplay()
        let keep = cachedMasks.firstIndex { CMTimeCompare($0.time, current.time) >= 0 } ?? cachedMasks.count
        let removed = cachedMasks.count - keep
        cachedMasks.removeSubrange(keep..<cachedMasks.count)
        // Invalidate metadata before touching disk: even if truncation fails,
        // rejected tail bytes can never be replayed as accepted masks.
        let prefix = cachedMasks.map {
            "\($0.subjectID.uuidString)|\($0.segment)|\($0.time.value)/\($0.time.timescale)|\($0.sha256)"
        }.joined(separator: "\n")
        correctionStart = current.time
        session?.reset()
        point = nil
        currentOverlay = nil
        extendCacheTimeline(to: current.time)
        report?.correctionsStarted += 1
        report?.pendingCorrectionFromSeconds = current.time.seconds
        let correction = VideoCorrectionRecord(subjectID: subjectID,
            fromPTSValue: current.time.value, fromPTSTimescale: current.time.timescale,
            retainedMaskFrames: keep, invalidatedMaskFrames: removed,
            retainedPrefixMetadataSHA256: digest(Data(prefix.utf8)))
        report?.recentCorrections.append(correction)
        if (report?.recentCorrections.count ?? 0) > 120 { report?.recentCorrections.removeFirst() }
        report?.lastState = nil
        report?.reachedEnd = false
        report?.lastResetReason = "manual correction; masks at/after current frame invalidated"
        report?.lastAction = "Awaiting same-subject reselection"
        refreshCacheReport()
        try reopenCacheWriter()
        return display(current, overlay: nil,
                       message: "Earlier masks kept. Seek to a clear view if needed, tap the same subject, then Continue same subject.")
    }

    func startNewSubject() throws -> OwnedVideoDisplay {
        guard let current else { throw OwnedTemporalError.invalid("Choose a video frame first.") }
        session?.reset()
        resetMaskCache(create: false)
        subjectID = nil
        correctionStart = nil
        point = nil
        currentOverlay = nil
        report?.subjectID = nil
        report?.pendingCorrectionFromSeconds = nil
        report?.lastState = nil
        report?.reachedEnd = false
        report?.lastResetReason = "explicit new subject; previous subject cache cleared"
        report?.lastAction = "Awaiting new subject selection"
        return display(current, overlay: nil, message: "Tap the new subject. Previous subject masks have been cleared.")
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
        stopReplay()
        currentOverlay = nil
        point = nil
        current = nil
        report?.seeks += 1
        report?.lastResetReason = "seek; masks and subject preserved, inference memory reset"
        report?.lastState = nil
        report?.reachedEnd = false
        do {
            let time = CMTime(seconds: seconds, preferredTimescale: 60000)
            try startReader(at: time)
            guard let frame = try readFrame(notBefore: time) else {
                throw OwnedTemporalError.invalid("No frame at this time. Seek earlier.")
            }
            current = frame
            currentOverlay = try cachedOverlay(at: frame.time)
            report?.lastAction = "Seek completed; cache preserved"
            let message = correctionStart != nil
                ? "Correction pending. Tap the same subject at this frame or seek to a clear view."
                : subjectID != nil
                    ? "Reviewing saved masks. Mark the first wrong frame with Correct tracking from here to resume tracking."
                    : "Tap the subject in this frame."
            return display(frame, overlay: currentOverlay, message: message)
        } catch {
            invalidatePrediction(error)
            throw error
        }
    }

    // Replay has its own decoder and reads only completed cache entries. The
    // inference decoder, current frame, point and temporal banks stay untouched.
    func beginReplay() throws {
        stopReplay()
        guard subjectID != nil, let start = cacheTimelineStart, cacheTimelineEnd != nil, let cacheDirectory else {
            throw OwnedTemporalError.invalid("Track some frames before replaying cached masks.")
        }
        do {
            let pair = try makeReader(at: start)
            replayReader = pair.0
            replayOutput = pair.1
            replayFile = try FileHandle(forReadingFrom: cacheDirectory.appendingPathComponent("masks.bin"))
            replayIndex = 0
            // Read report values before optional-chained mutation opens a
            // write access to the whole optional struct (Swift exclusivity).
            let predictionCount = report?.framesPredicted
            let state = report?.lastState
            report?.replayRuns += 1
            report?.replayError = nil
            report?.lastReplayPredictionCountBefore = predictionCount
            report?.lastReplayPredictionCountAfter = nil
            report?.lastReplayStateBefore = state
            report?.lastReplayStateAfter = nil
            report?.lastReplayFramesRead = 0
            report?.lastReplayMasksRead = 0
            report?.lastReplayUntrackedFrames = 0
            report?.lastReplaySubjectIDBefore = subjectID
            report?.lastReplaySubjectIDAfter = nil
        } catch {
            stopReplay()
            report?.replayError = error.localizedDescription
            throw error
        }
    }

    func nextReplay() throws -> OwnedVideoDisplay? {
        guard let replayReader, let replayOutput, let replayFile,
              let firstTime = cacheTimelineStart, let endTime = cacheTimelineEnd else {
            throw OwnedTemporalError.invalid("Cached replay is not open.")
        }
        if let replayLastTime, CMTimeCompare(replayLastTime, endTime) >= 0 { return nil }
        let start = now()
        do {
            return try autoreleasepool {
                try Task.checkCancellation()
                guard let frame = try readFrame(reader: replayReader, output: replayOutput,
                                                notBefore: replayLastTime == nil ? firstTime : nil,
                                                prepareModel: false) else {
                    throw OwnedTemporalError.invalid("Video ended before the cached timeline. Replay stopped.")
                }
                guard CMTimeCompare(frame.time, endTime) <= 0,
                      replayLastTime != nil || CMTimeCompare(frame.time, firstTime) == 0,
                      replayLastTime.map({ CMTimeCompare(frame.time, $0) > 0 }) ?? true else {
                    throw OwnedTemporalError.invalid("Decoded replay timestamps do not match the cached timeline.")
                }
                var overlay: CGImage?
                var hasMask = false
                if replayIndex < cachedMasks.count {
                    let entry = cachedMasks[replayIndex]
                    let order = CMTimeCompare(frame.time, entry.time)
                    guard order <= 0, entry.subjectID == subjectID else {
                        throw OwnedTemporalError.invalid("Cached mask ownership or timestamp mismatch. Replay stopped.")
                    }
                    if order == 0 {
                        overlay = try readCachedOverlay(index: replayIndex, file: replayFile)
                        replayIndex += 1
                        hasMask = true
                    }
                }
                try Task.checkCancellation()
                replayLastTime = frame.time
                report?.replayFrames += 1
                report?.lastReplayFramesRead += 1
                if hasMask { report?.lastReplayMasksRead += 1 }
                else { report?.lastReplayUntrackedFrames += 1 }
                report?.replayRequestTotals.add(elapsed(start))
                return display(frame, overlay: overlay,
                               message: hasMask
                                ? "Cached replay · \(replayIndex)/\(cachedMasks.count) masks · same subject"
                                : "Not tracked at this frame · no mask saved · same subject")
            }
        } catch {
            stopReplay()
            report?.replayError = error.localizedDescription
            throw error
        }
    }

    func finishReplay() -> OwnedVideoDisplay? {
        stopReplay()
        let predictionCount = report?.framesPredicted
        let state = report?.lastState
        report?.lastReplayPredictionCountAfter = predictionCount
        report?.lastReplayStateAfter = state
        report?.lastReplaySubjectIDAfter = subjectID
        guard let current else { return nil }
        return display(current, overlay: currentOverlay,
                       message: "Replay finished. Returned to your working frame.")
    }

    private func stopReplay() {
        replayReader?.cancelReading()
        replayReader = nil
        replayOutput = nil
        try? replayFile?.close()
        replayFile = nil
        replayIndex = 0
        replayLastTime = nil
    }

    private func readCachedOverlay(index: Int, file: FileHandle) throws -> CGImage {
        let entry = cachedMasks[index]
        guard entry.subjectID == subjectID else { throw OwnedTemporalError.invalid("Mask belongs to a different subject.") }
        try file.seek(toOffset: UInt64(index * Self.maskByteCount))
        guard let data = try file.read(upToCount: Self.maskByteCount),
              data.count == Self.maskByteCount, digest(data) == entry.sha256 else {
            throw OwnedTemporalError.invalid("Cached mask is missing or damaged.")
        }
        return try maskImage(packed: data)
    }

    private func cachedOverlay(at time: CMTime) throws -> CGImage? {
        guard let index = cachedMasks.firstIndex(where: { CMTimeCompare($0.time, time) == 0 }),
              let cacheDirectory else { return nil }
        let file = try FileHandle(forReadingFrom: cacheDirectory.appendingPathComponent("masks.bin"))
        defer { try? file.close() }
        return try readCachedOverlay(index: index, file: file)
    }

    private func extendCacheTimeline(to time: CMTime) {
        if cacheTimelineStart.map({ CMTimeCompare(time, $0) < 0 }) ?? true { cacheTimelineStart = time }
        if cacheTimelineEnd.map({ CMTimeCompare(time, $0) > 0 }) ?? true { cacheTimelineEnd = time }
    }

    private func refreshCacheReport() {
        report?.cachedMaskFrames = cachedMasks.count
        report?.cachedMaskBytes = cachedMasks.count * Self.maskByteCount
        report?.cacheSelectionSegment = cachedMasks.last?.segment
        report?.cachedStartSeconds = cachedMasks.first?.time.seconds
        report?.cachedEndSeconds = cachedMasks.last?.time.seconds
        report?.cacheTimelineStartSeconds = cacheTimelineStart?.seconds
        report?.cacheTimelineEndSeconds = cacheTimelineEnd?.seconds
        report?.cacheLimitReached = cachedMasks.count >= Self.maximumCachedFrames
    }

    private func reopenCacheWriter() throws {
        do {
            try cacheWriter?.close()
            cacheWriter = nil
            guard let cacheDirectory else { throw OwnedTemporalError.invalid("Mask cache unavailable. Start a new subject to retry.") }
            let file = try FileHandle(forWritingTo: cacheDirectory.appendingPathComponent("masks.bin"))
            do {
                let retainedBytes = UInt64(cachedMasks.count * Self.maskByteCount)
                guard try file.seekToEnd() >= retainedBytes else {
                    throw OwnedTemporalError.invalid("The retained mask prefix is incomplete. Cached replay cannot be trusted.")
                }
                try file.truncate(atOffset: retainedBytes)
                try file.seek(toOffset: retainedBytes)
                cacheWriter = file
                report?.cacheError = nil
            } catch {
                try? file.close()
                throw error
            }
        } catch {
            cacheWriter = nil
            report?.cacheError = error.localizedDescription
            throw error
        }
    }

    private func resetMaskCache(create: Bool) {
        stopReplay()
        try? cacheWriter?.close()
        cacheWriter = nil
        cachedMasks.removeAll()
        cacheTimelineStart = nil
        cacheTimelineEnd = nil
        report?.cachedMaskFrames = 0
        report?.cachedMaskBytes = 0
        report?.cacheSelectionSegment = nil
        report?.cachedStartSeconds = nil
        report?.cachedEndSeconds = nil
        report?.cacheTimelineStartSeconds = nil
        report?.cacheTimelineEndSeconds = nil
        report?.cacheLimitReached = false
        report?.cacheError = nil
        if let directory = cacheDirectory {
            do { try FileManager.default.removeItem(at: directory) }
            catch {
                // Keep the directory identity for another cleanup attempt;
                // don't accumulate more caches when deletion fails.
                report?.cacheError = error.localizedDescription
                return
            }
        }
        cacheDirectory = nil
        guard create else { return }
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("ByteWaveMaskReplay-\(UUID().uuidString)", isDirectory: true)
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            cacheDirectory = directory
            let file = directory.appendingPathComponent("masks.bin")
            guard FileManager.default.createFile(atPath: file.path, contents: nil) else {
                throw OwnedTemporalError.invalid("Could not create mask cache.")
            }
            cacheWriter = try FileHandle(forWritingTo: file)
        } catch {
            report?.cacheError = error.localizedDescription
        }
    }

    private func cacheMask(_ mask: OwnedTensor, at time: CMTime) {
        guard let cacheWriter, let subjectID else { return }
        guard cachedMasks.count < Self.maximumCachedFrames else {
            report?.cacheLimitReached = true
            return
        }
        let start = now()
        defer { report?.cacheWriteTotals.add(elapsed(start)) }
        do {
            guard mask.shape == [1, 1, 1024, 1024], mask.values.count == 1024 * 1024,
                  cachedMasks.last.map({ CMTimeCompare(time, $0.time) > 0 }) ?? true else {
                throw OwnedTemporalError.invalid("Invalid mask cache shape or timestamp order.")
            }
            // Lossless relative to the displayed binary mask, not to logits.
            var packed = [UInt8](repeating: 0, count: Self.maskByteCount)
            for i in mask.values.indices where mask.values[i] > 0 {
                packed[i >> 3] |= UInt8(1 << (i & 7))
            }
            let data = Data(packed)
            let hash = digest(data)
            try cacheWriter.write(contentsOf: data)
            // Publish only after the complete write succeeds. A failed partial
            // tail is never indexed/read, and no later appends are attempted.
            let segment = report?.segmentsStarted ?? 0
            cachedMasks.append(CachedMask(time: time, sha256: hash, subjectID: subjectID, segment: segment))
            extendCacheTimeline(to: time)
            refreshCacheReport()
        } catch {
            report?.cacheError = error.localizedDescription
            try? cacheWriter.close()
            self.cacheWriter = nil
        }
    }

    func saveReport() throws -> URL {
        guard var snapshot = report else { throw OwnedTemporalError.invalid("No video report yet.") }
        snapshot.thermalStateAtSnapshot = thermal()
        snapshot.cachedSegments = []
        for entry in cachedMasks {
            if let last = snapshot.cachedSegments.last, last.segment == entry.segment, last.subjectID == entry.subjectID {
                let index = snapshot.cachedSegments.count - 1
                snapshot.cachedSegments[index].masks += 1
                snapshot.cachedSegments[index].lastPTSValue = entry.time.value
                snapshot.cachedSegments[index].lastPTSTimescale = entry.time.timescale
            } else {
                snapshot.cachedSegments.append(VideoCachedSegmentRecord(subjectID: entry.subjectID, segment: entry.segment,
                    masks: 1, firstPTSValue: entry.time.value, firstPTSTimescale: entry.time.timescale,
                    lastPTSValue: entry.time.value, lastPTSTimescale: entry.time.timescale))
            }
        }
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
        resetMaskCache(create: false)
        subjectID = nil
        correctionStart = nil
        currentOverlay = nil
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
        reader?.cancelReading()
        let pair = try makeReader(at: time)
        reader = pair.0
        output = pair.1
    }

    private func makeReader(at time: CMTime) throws -> (AVAssetReader, AVAssetReaderTrackOutput) {
        guard let asset, let track else { throw OwnedTemporalError.invalid("No video is open.") }
        let nextReader = try AVAssetReader(asset: asset)
        let nextOutput = AVAssetReaderTrackOutput(track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        nextOutput.alwaysCopiesSampleData = false
        guard nextReader.canAdd(nextOutput) else { throw OwnedTemporalError.invalid("Cannot decode this video.") }
        nextReader.add(nextOutput)
        nextReader.timeRange = CMTimeRange(start: time, duration: CMTimeSubtract(duration, time))
        guard nextReader.startReading() else { throw nextReader.error ?? OwnedTemporalError.invalid("Video decoding could not start.") }
        return (nextReader, nextOutput)
    }

    private func readFrame(notBefore: CMTime? = nil) throws -> PreparedFrame? {
        guard let reader, let output else { throw OwnedTemporalError.invalid("The decoder is not ready.") }
        return try readFrame(reader: reader, output: output, notBefore: notBefore, prepareModel: true)
    }

    private func readFrame(reader: AVAssetReader, output: AVAssetReaderTrackOutput,
                           notBefore: CMTime?, prepareModel: Bool) throws -> PreparedFrame? {
        let start = now()
        var stages: [String: Double] = [:]
        while true {
            try Task.checkCancellation()
            var skippedPreroll = false
            // Temporary native buffers and rendering intermediates leave this pool.
            let decoded: PreparedFrame? = try autoreleasepool {
                let sampleStart = now()
                let nextSample = output.copyNextSampleBuffer()
                stages["sampleAcquisition", default: 0] += elapsed(sampleStart)
                guard let sample = nextSample else { return nil }
                let orientationStart = now()
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
                if prepareModel {
                    report?.lastDecodedSize = [CVPixelBufferGetWidth(buffer), CVPixelBufferGetHeight(buffer)]
                    report?.lastDisplaySize = [Double(bounds.width), Double(bounds.height)]
                }
                let normalized = presented.transformed(by: CGAffineTransform(translationX: -bounds.minX, y: -bounds.minY))
                let oriented = normalized.transformed(by: CGAffineTransform(a: 1, b: 0, c: 0, d: -1, tx: 0, ty: bounds.height))
                stages["orientationGraphSetup"] = elapsed(orientationStart)
                var input: CVPixelBuffer?
                let color = CGColorSpace(name: CGColorSpace.sRGB)!
                if prepareModel {
                    let allocationStart = now()
                    let code = CVPixelBufferCreate(kCFAllocatorDefault, 1024, 1024, kCVPixelFormatType_32BGRA,
                                                  [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &input)
                    guard code == kCVReturnSuccess, let input else { throw OwnedTemporalError.invalid("Could not allocate the model image.") }
                    stages["modelBufferAllocation"] = elapsed(allocationStart)
                    let renderStart = now()
                    let square = oriented.transformed(by: CGAffineTransform(scaleX: 1024 / bounds.width, y: 1024 / bounds.height))
                    context.render(square, to: input, bounds: CGRect(x: 0, y: 0, width: 1024, height: 1024), colorSpace: color)
                    stages["modelInputRender"] = elapsed(renderStart)
                }
                let previewStart = now()
                let scale = min(1, 1024 / max(bounds.width, bounds.height))
                let preview = oriented.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
                // Finish preview rendering on this actor. Sharing the immutable
                // CGImage avoids compressing and decoding a PNG for every frame.
                guard let image = context.createCGImage(preview, from: preview.extent, format: .RGBA8,
                                                       colorSpace: color, deferred: false) else {
                    throw OwnedTemporalError.invalid("Could not prepare the video preview.")
                }
                stages["previewImageCreation"] = elapsed(previewStart)
                return PreparedFrame(buffer: input, preview: image, time: time,
                                     preparationMS: elapsed(start), preparationStages: stages)
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
        guard let current, let buffer = current.buffer, let session, let point, let subjectID else { throw OwnedTemporalError.invalid("Select a subject first.") }
        do {
            return try autoreleasepool {
                let start = now()
                let prediction = try session.predict(image: buffer, at: current.time, initialPoint: point) { component in
                    try checkpoint("Before prediction: \(component)", time: current.time)
                }
                let predictionMS = elapsed(start)
                guard let mask = prediction.tensors["low_res_mask"],
                      let previewMask = prediction.tensors["high_res_mask"], let score = prediction.tensors["object_score"],
                      let iou = prediction.tensors["best_iou"] else { throw OwnedTemporalError.invalid("Incomplete prediction.") }
                let maskStart = now()
                let overlay = try maskImage(previewMask)
                let maskMS = elapsed(maskStart)
                try Task.checkCancellation()
                cacheMask(previewMask, at: current.time)
                currentOverlay = overlay
                let fraction = Double(mask.values.lazy.filter { $0 > 0 }.count) / Double(mask.values.count)
                let row = VideoFrameRecord(requestIndex: report?.framesPredicted ?? 0, subjectID: subjectID, segment: report?.segmentsStarted ?? 0,
                    ptsValue: current.time.value, ptsTimescale: current.time.timescale, initialized: initialized,
                    foregroundFraction: fraction, objectScore: score.values[0], predictedIoU: iou.values[0],
                    modelMilliseconds: prediction.modelMilliseconds,
                    sessionStageMilliseconds: prediction.sessionStageMilliseconds,
                    decodeAndPreparationMilliseconds: preparationMS,
                    preparationStageMilliseconds: initialized ? [:] : current.preparationStages,
                    predictionAndStateMilliseconds: predictionMS, maskRenderingMilliseconds: maskMS,
                    requestMilliseconds: elapsed(requestStart), state: prediction.state)
                report?.framesPredicted += 1
                report?.lastState = prediction.state
                if fraction == 0 { report?.emptyMasks += 1 }
                report?.recentFrames.append(row)
                if (report?.recentFrames.count ?? 0) > 120 { report?.recentFrames.removeFirst() }
                if !initialized {
                    for (name, ms) in prediction.modelMilliseconds { report?.warmModelTotals[name, default: VideoTimingTotal()].add(ms) }
                    for (name, ms) in prediction.sessionStageMilliseconds {
                        report?.warmSessionStageTotals[name, default: VideoTimingTotal()].add(ms)
                    }
                    report?.warmRequestTotals.add(row.requestMilliseconds)
                    report?.warmPredictionAndStateTotals.add(predictionMS)
                    report?.warmDecodeAndPreparationTotals.add(preparationMS)
                    for (name, ms) in row.preparationStageMilliseconds {
                        report?.warmPreparationStageTotals[name, default: VideoTimingTotal()].add(ms)
                    }
                    report?.warmMaskRenderingTotals.add(maskMS)
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
        stopReplay()
        currentOverlay = nil
        point = nil
        report?.lastState = nil
        report?.lastResetReason = "error or cancellation"
        report?.lastError = error.localizedDescription
        report?.lastAction = "Stopped; saved masks retained; correction or new subject required"
    }

    private func display(_ frame: PreparedFrame, overlay: CGImage?, message: String) -> OwnedVideoDisplay {
        OwnedVideoDisplay(previewImage: frame.preview, overlayImage: overlay, seconds: frame.time.seconds,
                          durationSeconds: duration.seconds, message: message, cachedFrames: cachedMasks.count,
                          cacheNotice: report?.cacheError.map { "Mask cache: \($0)" }
                              ?? (report?.cacheLimitReached == true ? "Mask cache full. Uncached frames cannot replay with a mask." : nil),
                          hasSubject: subjectID != nil, isTracking: point != nil,
                          correctionFromSeconds: correctionStart?.seconds,
                          canReplay: subjectID != nil && cacheDirectory != nil && cacheTimelineStart != nil && cacheTimelineEnd != nil)
    }
    private func check(_ ticket: UInt64) throws {
        try Task.checkCancellation()
        guard ticket == generation else { throw CancellationError() }
    }
    private func checkpoint(_ stage: String, time: CMTime? = nil) throws {
        guard let checkpointURL else { return }
        let data: [String: Any] = ["schema": "bytewave.video-checkpoint.v1", "stage": stage,
            "modelVariant": "memoryfp16projection", "fixtureSHA256": report?.fixtureSHA256 ?? "",
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
    private func maskImage(_ mask: OwnedTensor) throws -> CGImage {
        // Core ML already produced/validated the upsampled logits. Threshold
        // those, rather than magnifying a thresholded 256-square bitmap.
        // OwnedTensor validates finiteness; do not change model/state tensors.
        guard mask.shape == [1, 1, 1024, 1024], mask.values.count == 1024 * 1024 else {
            throw OwnedTemporalError.invalid("Unexpected preview mask shape.")
        }
        let side = 1024
        var bytes = [UInt8](repeating: 0, count: side * side * 4)
        for i in mask.values.indices where mask.values[i] > 0 {
            bytes[i * 4] = 90; bytes[i * 4 + 1] = 160; bytes[i * 4 + 2] = 255; bytes[i * 4 + 3] = 255
        }
        return try overlayImage(bytes: bytes)
    }

    private func maskImage(packed: Data) throws -> CGImage {
        guard packed.count == Self.maskByteCount else {
            throw OwnedTemporalError.invalid("Unexpected cached mask size.")
        }
        var bytes = [UInt8](repeating: 0, count: 1024 * 1024 * 4)
        for (byteIndex, bits) in packed.enumerated() where bits != 0 {
            for bit in 0..<8 where bits & UInt8(1 << bit) != 0 {
                let i = (byteIndex * 8 + bit) * 4
                bytes[i] = 90; bytes[i + 1] = 160; bytes[i + 2] = 255; bytes[i + 3] = 255
            }
        }
        return try overlayImage(bytes: bytes)
    }

    private func overlayImage(bytes: [UInt8]) throws -> CGImage {
        let side = 1024
        // The provider owns its Data independently of Core ML output lifetimes.
        // Pass the image directly: a larger mask must not add a PNG round-trip.
        guard let provider = CGDataProvider(data: Data(bytes) as CFData),
              let color = CGColorSpace(name: CGColorSpace.sRGB),
              let image = CGImage(width: side, height: side, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: side * 4,
                  space: color, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                  provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent) else {
            throw OwnedTemporalError.invalid("Could not draw mask.")
        }
        return image
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
    @State private var cachedFrames = 0
    @State private var cacheNotice: String?
    @State private var replaying = false
    @State private var hasSubject = false
    @State private var correctionFromSeconds: Double?
    @State private var canReplay = false
    @State private var showNewSubjectConfirmation = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                PhotosPicker("Choose video", selection: $selection, matching: .videos).disabled(busy)
                Text("Memory FP16 + fused encoder · CPU + GPU").font(.subheadline).foregroundStyle(.secondary)
                if let preview {
                    GeometryReader { geometry in
                        ZStack(alignment: .topLeading) {
                            Image(uiImage: preview).resizable()
                            if showMask, let overlay { Image(uiImage: overlay).resizable().interpolation(.low).opacity(0.45) }
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
                            guard !hasSubject || correctionFromSeconds != nil else {
                                status = "Use Correct tracking from here to reselect this subject, or New subject to replace it."
                                return
                            }
                            point = CGPoint(x: min(max(location.x / geometry.size.width, 0), 1),
                                            y: min(max(location.y / geometry.size.height, 0), 1))
                            tracking = false
                            overlay = nil
                            reportURL = nil
                            status = hasSubject ? "Tap Continue same subject to resume from this selection." : "Tap Track selected subject to begin."
                        }
                        .accessibilityLabel("Current video frame. Tap the subject to select it.")
                    }.aspectRatio(preview.size.width / preview.size.height, contentMode: .fit)
                    Button("Use center point") {
                        point = CGPoint(x: 0.5, y: 0.5); tracking = false; overlay = nil; reportURL = nil
                    }.disabled(busy || (hasSubject && correctionFromSeconds == nil))
                    Button(hasSubject ? "Continue same subject" : "Track selected subject") { initialize() }
                        .buttonStyle(.borderedProminent)
                        .disabled(busy || point == nil || tracking || (hasSubject && correctionFromSeconds == nil)
                                  || correctionFromSeconds.map({ seconds < $0 }) == true)
                    if hasSubject {
                        Button("Correct tracking from here") { correctTracking() }
                            .buttonStyle(.bordered)
                            .disabled(busy || correctionFromSeconds != nil)
                        if let correctionFromSeconds {
                            Text(String(format: "Correction starts at %.3f s. Seek to a clear view at this time or later, then tap the same subject. The gap stays untracked.", correctionFromSeconds))
                                .font(.footnote)
                        } else {
                            Text("Pause and seek to the first wrong frame before correcting. Earlier masks are kept; masks from that frame onward are removed.")
                                .font(.footnote)
                        }
                        Button("New subject…") { showNewSubjectConfirmation = true }.disabled(busy)
                    }
                    HStack {
                        Button("Next frame") { advance(continuous: false) }.disabled(busy || !tracking || ended)
                        Button(playing ? "Pause" : "Run tracking") {
                            if playing { pauseRequested = true } else { advance(continuous: true) }
                        }.disabled(playing ? pauseRequested : (busy || !tracking || ended))
                    }.buttonStyle(.bordered)
                    Button(replaying ? "Stop replay" : "Replay cached masks") {
                        if replaying { pauseRequested = true } else { replay() }
                    }
                    .buttonStyle(.bordered)
                    .disabled(replaying ? pauseRequested : (busy || !canReplay))
                    if cachedFrames > 0 {
                        Text("\(cachedFrames) masks saved for this subject").font(.footnote)
                    }
                    if let cacheNotice { Text(cacheNotice).font(.footnote).foregroundStyle(.secondary) }
                    Toggle("Show mask", isOn: $showMask)
                }
                if duration > 0 {
                    Text(String(format: "Frame %.3f s / %.3f s", seconds, duration)).monospacedDigit()
                    Slider(value: $seekSeconds, in: 0...max(duration - 0.001, 0.001)).disabled(busy)
                    HStack {
                        Button(String(format: "Seek to %.2f s", seekSeconds)) { seek(to: seekSeconds) }
                        Button("Go to start") { seek(to: 0) }
                    }.disabled(busy)
                    Text("Seeking keeps saved masks and pauses tracking. Use Correct tracking from here to resume the same subject, or Replay cached masks to review it.").font(.footnote)
                }
                if busy { ProgressView() }
                Text(status).textSelection(.enabled)
                if let reportURL { ShareLink("Share video tracking report", item: reportURL).disabled(busy) }
                Text("Frames advance as processing finishes. Audio is off. Check that the blue mask follows your subject.")
                    .font(.footnote).foregroundStyle(.secondary)
            }.padding()
        }
        .navigationTitle("Track a video")
        .confirmationDialog("Start a new subject?", isPresented: $showNewSubjectConfirmation, titleVisibility: .visible) {
            Button("Clear masks and start new subject", role: .destructive) { newSubject() }
            Button("Cancel", role: .cancel) { }
        } message: {
            Text("This clears the current subject's saved masks. Use Correct tracking from here to keep the same subject and its earlier masks.")
        }
        .onChange(of: selection) { _, item in open(item) }
        .onChange(of: scenePhase) { _, phase in
            if phase != .active { pauseRequested = true }
        }
        .onDisappear {
            lifecycle = UUID()
            job?.cancel()
            busy = false
            playing = false
            replaying = false
            cachedFrames = 0
            cacheNotice = nil
            hasSubject = false
            correctionFromSeconds = nil
            canReplay = false
            showNewSubjectConfirmation = false
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
        guard let item, !busy, let root = Bundle.main.resourceURL?.appendingPathComponent("DeviceValidationData/memoryfp16projection") else { return }
        guard FileManager.default.fileExists(atPath: root.appendingPathComponent("fixture.json").path) else {
            status = "The fused encoder fixture is missing. Run the Mac preparation command, then rebuild."
            return
        }
        let ticket = lifecycle
        busy = true; preview = nil; overlay = nil; point = nil; tracking = false; ended = false
        cachedFrames = 0; cacheNotice = nil
        hasSubject = false; correctionFromSeconds = nil; canReplay = false
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
            ended = false
        }
    }

    private func correctTracking() {
        guard !busy, hasSubject, correctionFromSeconds == nil else { return }
        point = nil; overlay = nil; tracking = false; ended = false
        perform { apply(try await runner.correctFromCurrentFrame()) }
    }

    private func newSubject() {
        guard !busy else { return }
        point = nil; overlay = nil; tracking = false; ended = false
        perform { apply(try await runner.startNewSubject()) }
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

    private func replay() {
        guard !busy, canReplay else { return }
        let ticket = lifecycle
        busy = true
        replaying = true
        pauseRequested = false
        reportURL = nil
        point = nil
        job = Task {
            defer { if ticket == lifecycle { busy = false; replaying = false } }
            var failure: String?
            do {
                try Task.checkCancellation()
                try await runner.beginReplay()
                var firstPTS: Double?
                var wallStart = 0.0
                while !pauseRequested {
                    try Task.checkCancellation()
                    guard let frame = try await runner.nextReplay() else { break }
                    try Task.checkCancellation()
                    guard ticket == lifecycle else { return }
                    if firstPTS == nil {
                        firstPTS = frame.seconds
                        wallStart = ProcessInfo.processInfo.systemUptime
                    }
                    let deadline = wallStart + frame.seconds - (firstPTS ?? frame.seconds)
                    // No dropped frames. Slow decode/render makes replay late;
                    // the processing report excludes this presentation pacing.
                    while !pauseRequested {
                        let remaining = deadline - ProcessInfo.processInfo.systemUptime
                        if remaining <= 0 { break }
                        try await Task.sleep(for: .seconds(min(remaining, 0.05)))
                    }
                    if pauseRequested { break }
                    apply(frame)
                    await Task.yield()
                }
            } catch {
                failure = error.localizedDescription
            }
            guard ticket == lifecycle, !Task.isCancelled else { return }
            // Restore the inference position before allowing selection/Next.
            // Displaying an old cached frame must never imply old model state.
            let restored = await runner.finishReplay()
            guard ticket == lifecycle, !Task.isCancelled else { return }
            if let restored { apply(restored) }
            if let failure { status = "Replay stopped: \(failure) Returned to your working frame." }
            do {
                let saved = try await runner.saveReport()
                if ticket == lifecycle, !Task.isCancelled { reportURL = saved }
            } catch {
                if ticket == lifecycle, !Task.isCancelled { status += " Report could not be saved: \(error.localizedDescription)" }
            }
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
                canReplay = false
                status = "\(error.localizedDescription) Seek to a frame to recover; saved earlier masks are retained."
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
        preview = UIImage(cgImage: result.previewImage, scale: 1, orientation: .up)
        overlay = result.overlayImage.map { UIImage(cgImage: $0, scale: 1, orientation: .up) }
        seconds = result.seconds
        seekSeconds = min(result.seconds, max(result.durationSeconds - 0.001, 0))
        duration = result.durationSeconds
        status = result.message
        cachedFrames = result.cachedFrames
        cacheNotice = result.cacheNotice
        hasSubject = result.hasSubject
        tracking = result.isTracking
        correctionFromSeconds = result.correctionFromSeconds
        canReplay = result.canReplay
    }
    private func hardware() -> String {
        var size: size_t = 0
        guard sysctlbyname("hw.machine", nil, &size, nil, 0) == 0, size > 0 else { return "Unknown" }
        var bytes = [CChar](repeating: 0, count: size)
        guard sysctlbyname("hw.machine", &bytes, &size, nil, 0) == 0 else { return "Unknown" }
        return String(cString: bytes)
    }
}


