import SwiftUI
import CoreML
import UIKit
import Darwin

@main
struct SegmentationProbeApp: App {
    var body: some Scene {
        WindowGroup { ProbeView() }
    }
}

enum ProbeMode: String, CaseIterable, Identifiable, Sendable {
    case neuralEngine = "CPU + Neural Engine"
    case gpu = "CPU + GPU"
    case automatic = "Automatic"

    var id: String { rawValue }

    var computeUnits: MLComputeUnits {
        switch self {
        case .neuralEngine: return .cpuAndNeuralEngine
        case .gpu: return .cpuAndGPU
        case .automatic: return .all
        }
    }
}

struct OperationPlacement: Codable, Sendable {
    let path: String
    let operation: String
    let outputs: [String]
    let preferredDevice: String?
    let supportedDevices: [String]
}

struct ComponentReport: Codable, Sendable {
    let model: String
    var compileSeconds: Double?
    var loadSeconds: Double?
    var loadSucceeded = false
    var planSucceeded = false
    var errors: [String: String] = [:]
    var placements: [OperationPlacement] = []

    var summary: String {
        let counts = Dictionary(grouping: placements, by: { $0.preferredDevice ?? "unreported" })
            .mapValues { $0.count }
        let distribution = counts.keys.sorted().map { "\($0): \(counts[$0]!)" }.joined(separator: ", ")
        let title = model.replacingOccurrences(of: "EdgeTAMVideo", with: "")
        let outcome = "\(title): load \(loadSucceeded ? "OK" : "FAILED"), plan \(planSucceeded ? "OK" : "FAILED")"
        let details = errors.keys.sorted().map { "\($0): \(errors[$0]!)" }
        return ([outcome, distribution] + details).filter { !$0.isEmpty }.joined(separator: "\n")
    }
}

struct ProbeReport: Codable, Sendable {
    let schema = "bytewave.segmentation-placement-probe.v1"
    let generatedAt: Date
    let hardware: String
    let operatingSystem: String
    let mode: String
    let modelRevision = "6bfdd4765e42508c7707566fff52e65add8b8e3a"
    let availableComputeDevices: [String]
    let measurementScope = "Model compilation, loading, and planned operation placement only. No predictions, tracking, mask-quality assessment, playback benchmark, or runtime Neural Engine measurement were performed."
    let components: [ComponentReport]
}

actor ModelInspector {
    static let names = [
        "EdgeTAMVideoImageEncoder", "EdgeTAMVideoInitializer",
        "EdgeTAMVideoMemoryEncoder", "EdgeTAMVideoPropagator"
    ]

    func run(mode: ProbeMode, modelsDirectory: URL, hardware: String,
             progress: @Sendable (String) async -> Void) async -> ProbeReport {
        var components: [ComponentReport] = []
        for name in Self.names {
            await progress("Inspecting \(name.replacingOccurrences(of: "EdgeTAMVideo", with: ""))…")
            let url = modelsDirectory.appendingPathComponent(name + ".mlpackage")
            components.append(await inspect(url: url, name: name, mode: mode))
        }
        return ProbeReport(
            generatedAt: Date(), hardware: hardware,
            operatingSystem: ProcessInfo.processInfo.operatingSystemVersionString,
            mode: mode.rawValue,
            availableComputeDevices: MLComputeDevice.allComputeDevices.map(deviceName),
            components: components
        )
    }

    private func inspect(url: URL, name: String, mode: ProbeMode) async -> ComponentReport {
        var report = ComponentReport(model: name)
        guard FileManager.default.fileExists(atPath: url.path) else {
            report.errors["package"] = "Missing bundled model package: \(name).mlpackage"
            return report
        }
        let compiled: URL
        do {
            let start = ProcessInfo.processInfo.systemUptime
            compiled = try MLModel.compileModel(at: url)
            report.compileSeconds = ProcessInfo.processInfo.systemUptime - start
        } catch {
            report.errors["compile"] = describe(error)
            return report
        }
        // compileModel returns a temporary compiled model. Model instances are
        // released by loadForInspection before this directory is removed.
        defer { try? FileManager.default.removeItem(at: compiled) }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = mode.computeUnits
        do {
            report.loadSeconds = try await loadForInspection(compiled, configuration: configuration)
            report.loadSucceeded = true
        } catch {
            report.errors["load"] = describe(error)
        }
        // Try the plan even if model loading failed. Preserve both errors.
        do {
            let plan = try await MLComputePlan.load(contentsOf: compiled, configuration: configuration)
            guard case let .program(program) = plan.modelStructure else {
                report.errors["plan"] = "Expected an ML Program model."
                return report
            }
            for functionName in program.functions.keys.sorted() {
                guard let function = program.functions[functionName] else { continue }
                collect(function.block, path: functionName, plan: plan, into: &report.placements)
            }
            report.planSucceeded = true
        } catch {
            report.errors["plan"] = describe(error)
        }
        return report
    }

    private func loadForInspection(_ url: URL, configuration: MLModelConfiguration) async throws -> Double {
        let start = ProcessInfo.processInfo.systemUptime
        let model = try await MLModel.load(contentsOf: url, configuration: configuration)
        let elapsed = ProcessInfo.processInfo.systemUptime - start
        // Access its description to confirm a usable instance was returned.
        _ = model.modelDescription.inputDescriptionsByName
        return elapsed
    }

    private func collect(_ block: MLModelStructure.Program.Block, path: String,
                         plan: MLComputePlan, into rows: inout [OperationPlacement]) {
        for (index, operation) in block.operations.enumerated() {
            let operationPath = "\(path)/\(index)"
            let usage = plan.deviceUsage(for: operation)
            rows.append(OperationPlacement(
                path: operationPath, operation: operation.operatorName,
                outputs: operation.outputs.map { $0.name },
                preferredDevice: usage.map { deviceName($0.preferred) },
                supportedDevices: usage?.supported.map(deviceName) ?? []
            ))
            for (blockIndex, nested) in operation.blocks.enumerated() {
                collect(nested, path: "\(operationPath)/block\(blockIndex)", plan: plan, into: &rows)
            }
        }
    }

    private func deviceName(_ device: MLComputeDevice) -> String {
        switch device {
        case .cpu(_): return "CPU"
        case .gpu(_): return "GPU"
        case .neuralEngine(_): return "Neural Engine"
        @unknown default: return "Unknown"
        }
    }

    private func describe(_ error: Error) -> String {
        let error = error as NSError
        return "\(error.domain) [\(error.code)]: \(error.localizedDescription)"
    }
}

@MainActor
struct ProbeView: View {
    @State private var mode: ProbeMode = .neuralEngine
    @State private var running = false
    @State private var status = "Ready"
    @State private var summary = ""
    @State private var reportURL: URL?
    private let inspector = ModelInspector()

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    Text("Check where the four video models are planned to run.")
                    Picker("Compute devices", selection: $mode) {
                        ForEach(ProbeMode.allCases) { mode in Text(mode.rawValue).tag(mode) }
                    }
                    .pickerStyle(.menu)
                    .disabled(running)
                    Button(running ? "Inspecting…" : "Inspect models", action: start)
                        .buttonStyle(.borderedProminent)
                        .disabled(running)
                    if running { ProgressView() }
                    Text(status).accessibilityAddTraits(.updatesFrequently)
                    if !summary.isEmpty {
                        Text(summary).font(.system(.body, design: .monospaced)).textSelection(.enabled)
                    }
                    if let reportURL { ShareLink("Share report", item: reportURL) }
                    Text("CPU + Neural Engine still allows CPU fallback. Placement is a plan, not a measurement of actual execution. This probe does not run predictions or measure video FPS.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
                .padding()
            }
            .navigationTitle("Segmentation Probe")
            .tint(.blue)
        }
    }

    private func start() {
        #if targetEnvironment(simulator)
        status = "Select a physical iPhone in Xcode. Simulator results cannot answer the Neural Engine question."
        return
        #else
        guard let models = Bundle.main.resourceURL?.appendingPathComponent("Models") else {
            status = "Missing Models resource directory."
            return
        }
        running = true
        summary = ""
        reportURL = nil
        let selectedMode = mode
        let hardware = hardwareIdentifier()
        Task {
            let report = await inspector.run(mode: selectedMode, modelsDirectory: models, hardware: hardware) { message in
                await MainActor.run { status = message }
            }
            summary = report.components.map(\.summary).joined(separator: "\n\n")
            do {
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                encoder.dateEncodingStrategy = .iso8601
                let data = try encoder.encode(report)
                let documents = try FileManager.default.url(for: .documentDirectory, in: .userDomainMask,
                                                           appropriateFor: nil, create: true)
                let suffix: String
                switch selectedMode {
                case .neuralEngine: suffix = "cpu-neural-engine"
                case .gpu: suffix = "cpu-gpu"
                case .automatic: suffix = "automatic"
                }
                let url = documents.appendingPathComponent("segmentation-\(suffix)-\(UUID().uuidString).json")
                try data.write(to: url, options: .atomic)
                reportURL = url
                status = "Finished. Share the report for analysis."
            } catch {
                status = "Inspection finished, but saving the report failed: \(error.localizedDescription)"
            }
            running = false
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
