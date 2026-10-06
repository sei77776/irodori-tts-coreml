// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "IrodoriTTS",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        // BeatriceVC (experimental real-time voice conversion) ships in the same product so the
        // sample app can import it without another package product reference.
        .library(name: "IrodoriTTS", targets: ["IrodoriTTS", "BeatriceVC"]),
        .executable(name: "irodori", targets: ["IrodoriCLI"]),
    ],
    targets: [
        .target(name: "IrodoriNative", publicHeadersPath: "include",
                cxxSettings: [.define("IRODORI_COREML_ONLY", to: "1")],
                linkerSettings: [.linkedFramework("Foundation"), .linkedFramework("CoreML"),
                                 .linkedFramework("Accelerate")]),
        .target(name: "IrodoriTTS", dependencies: ["IrodoriNative"],
                linkerSettings: [.linkedFramework("AVFoundation")]),
        .executableTarget(name: "IrodoriCLI", dependencies: ["IrodoriTTS"]),
        .target(name: "BeatriceVC",
                linkerSettings: [.linkedFramework("AVFoundation"), .linkedFramework("CoreML"),
                                 .linkedFramework("Accelerate")]),
        .testTarget(name: "IrodoriTTSTests", dependencies: ["IrodoriTTS"]),
        .testTarget(name: "BeatriceVCTests", dependencies: ["BeatriceVC"],
                    resources: [.copy("Golden")]),
    ],
    cxxLanguageStandard: .cxx17
)
