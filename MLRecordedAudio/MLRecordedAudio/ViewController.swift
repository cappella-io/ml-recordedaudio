//
//  ViewController.swift
//  MLRecordedAudio
//
//  Created by Arshad Awati on 18/06/24.
//

import UIKit
import AVFoundation
import CoreML
import SoundAnalysis

class ViewController: UIViewController {
    @IBOutlet var testFrequencyTextFiled: UITextField!
    @IBOutlet var detectionSensitivityTextField: UITextField!
    @IBOutlet var cryEndThresholdTextField: UITextField!
    @IBOutlet var cryTranslationLength: UITextField!
    @IBOutlet var startDetectionButton: UIButton!
    @IBOutlet var cryOutputLabel: UILabel!
    @IBOutlet var cryStatusLabel: UILabel!
    private var babyCryDetector: BabyCryDetector?
    private let statusLabel = UILabel()
    var isDetecting = false
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
    }
    
    func setupBabyCryDetector(cryTranslationLength: Double, testFrequency: Double, detectionSensitivity: Int, cryEndThreshold: Int) {
        do {
                let melSpectrogramModel = try wave_to_logmel(configuration: MLModelConfiguration())
                let babyCryDetectionModel = try encodedBabyCryDetectionModel(configuration: MLModelConfiguration())
                babyCryDetector = BabyCryDetector(
                           melSpectrogramModel: melSpectrogramModel.model,
                           babyCryDetectionModel: babyCryDetectionModel.model,
                           viewController: self,
                           cryTranslationLength: cryTranslationLength,
                           testFrequency: testFrequency,
                           detectionSensitivity: detectionSensitivity,
                           cryEndThreshold: cryEndThreshold
                       )
                       babyCryDetector?.startDetection()
            } catch {
                fatalError("Failed to load models: \(error.localizedDescription)")
            }
        }
    
    private func setupUI() {
        view.backgroundColor = .white
        
        statusLabel.text = "Listening for baby cries..."
        statusLabel.textAlignment = .center
        statusLabel.frame = view.bounds
        statusLabel.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        
        view.addSubview(statusLabel)
    }
    
    func updateStatusLabel(with text: String) {
        print("Baby Cry Status:", text)
        DispatchQueue.main.async {
            self.statusLabel.text = text
        }
    }
    
    func handleBabyCryDetected(audioChunk: [Float]) {
            // Process or store the audio chunk as needed
            print("Audio chunk captured: \(audioChunk.count) samples")
            // Convert audio chunk to .wav file
            if let audioFileURL = saveAudioChunkAsWav(audioChunk: audioChunk) {
                print("Filename:", audioFileURL)
                
//                uploadAudioFile(url: audioFileURL) call the actual API for sending audio to inference
                    let fileManager = FileManager.default
                        let documentsURL = fileManager.urls(for: .documentDirectory, in: .userDomainMask)[0]
                        let fileURL = documentsURL.appendingPathComponent("babyCry.wav")

                       let activityViewController = UIActivityViewController(activityItems: [fileURL], applicationActivities: nil)
                        activityViewController.popoverPresentationController?.sourceView = self.view // For iPad compatibility

                        present(activityViewController, animated: true, completion: nil)
            }
        }
    
    private func saveAudioChunkAsWav(audioChunk: [Float]) -> URL? {
           let sampleRate: Float64 = 22050.0
           let numChannels: UInt32 = 1
           let bitsPerChannel: UInt32 = 16
           
           let fileManager = FileManager.default
           let documentsURL = fileManager.urls(for: .documentDirectory, in: .userDomainMask)[0]
           let fileURL = documentsURL.appendingPathComponent("babyCry.wav")
           
        var outputFormat = AudioStreamBasicDescription(
                    mSampleRate: sampleRate,
                    mFormatID: kAudioFormatLinearPCM,
                    mFormatFlags: kAudioFormatFlagIsSignedInteger | kAudioFormatFlagIsPacked,
                    mBytesPerPacket: numChannels * (bitsPerChannel / 8),
                    mFramesPerPacket: 1,
                    mBytesPerFrame: numChannels * (bitsPerChannel / 8),
                    mChannelsPerFrame: numChannels,
                    mBitsPerChannel: bitsPerChannel,
                    mReserved: 0
                )
        
           var audioFile: ExtAudioFileRef?
           var status = ExtAudioFileCreateWithURL(fileURL as CFURL, kAudioFileWAVEType, &outputFormat, nil, AudioFileFlags.eraseFile.rawValue, &audioFile)
           if status != noErr {
               print("Error creating audio file: \(status)")
               return nil
           }
           
           let bufferLength = audioChunk.count * MemoryLayout<Int16>.size
           var convertedData = Data(count: bufferLength)
           convertedData.withUnsafeMutableBytes { (convertedBuffer: UnsafeMutableRawBufferPointer) in
               let convertedPointer = convertedBuffer.bindMemory(to: Int16.self).baseAddress!
               for (index, sample) in audioChunk.enumerated() {
                   let intSample = Int16(sample * Float(Int16.max))
                   convertedPointer[index] = intSample
               }
           }
           
           let buffer = convertedData.withUnsafeBytes { (bufferPointer: UnsafeRawBufferPointer) in
               return AudioBuffer(mNumberChannels: numChannels, mDataByteSize: UInt32(bufferLength), mData: UnsafeMutableRawPointer(mutating: bufferPointer.baseAddress!))
           }
           
           var bufferList = AudioBufferList(mNumberBuffers: 1, mBuffers: buffer)
           status = ExtAudioFileWrite(audioFile!, UInt32(audioChunk.count), &bufferList)
           if status != noErr {
               print("Error writing audio file: \(status)")
               return nil
           }
           
           ExtAudioFileDispose(audioFile!)
           return fileURL
       }
    
    @IBAction func startDetectionButtonClicked(_ sender: Any) {
        if isDetecting {
                    // Stop detection
                    babyCryDetector?.stopDetection()
                    startDetectionButton.setTitle("Start Detection", for: .normal)
                    statusLabel.text = "Not listening"
                    isDetecting = false
                } else {
                    // Start detection
                    if testFrequencyTextFiled.text == nil || testFrequencyTextFiled.text!.isEmpty ||
                       detectionSensitivityTextField.text == nil || detectionSensitivityTextField.text!.isEmpty ||
                       cryEndThresholdTextField.text == nil || cryEndThresholdTextField.text!.isEmpty ||
                       cryTranslationLength.text == nil || cryTranslationLength.text!.isEmpty {
                        let alert = UIAlertController(title: "Cappella", message: "All fields are mandatory.", preferredStyle: .alert)
                        alert.addAction(UIAlertAction(title: "Okay", style: .default, handler: nil))
                        present(alert, animated: true, completion: nil)
                    } else {
                        if let testFrequency = Double(testFrequencyTextFiled.text!),
                           let detectionSensitivity = Int(detectionSensitivityTextField.text!),
                           let cryEndThreshold = Int(cryEndThresholdTextField.text!),
                           let cryTranslationLength = Double(cryTranslationLength.text!) {
                            setupUI()
                            setupBabyCryDetector(
                                cryTranslationLength: cryTranslationLength,
                                testFrequency: testFrequency,
                                detectionSensitivity: detectionSensitivity,
                                cryEndThreshold: cryEndThreshold
                            )
                            babyCryDetector?.startDetection()
                            startDetectionButton.setTitle("Stop Detection", for: .normal)
                            isDetecting = true
                        } else {
                            let alert = UIAlertController(title: "Cappella", message: "Invalid input values.", preferredStyle: .alert)
                            alert.addAction(UIAlertAction(title: "Okay", style: .default, handler: nil))
                            present(alert, animated: true, completion: nil)
                        }
                    }
                }
            }
}

class BabyCryDetector {
    private let audioEngine = AVAudioEngine()
    private let inputNode: AVAudioInputNode
    private let bufferSize: AVAudioFrameCount
    private let melSpectrogramModel: MLModel
    private let babyCryDetectionModel: MLModel
    private let bufferQueue: CircularBuffer<Float>
    private let sampleRate: Double
    private let cryTranslationLength: Double
    private let secondaryBufferQueue: CircularBuffer<Float> //additional buffer
    private let testFrequency: Double
    private weak var viewController: ViewController?
    private var detectInRow: Int = 0
    private var noCryInRow: Int = 0
    private let detectionSensitivity: Int
    private let cryEndThreshold: Int

    private var processingTimer: Timer?
    private var detectionStartIndex: Int?


    init(melSpectrogramModel: MLModel, babyCryDetectionModel: MLModel, viewController: ViewController, cryTranslationLength: Double, testFrequency: Double, detectionSensitivity: Int, cryEndThreshold: Int) {
        self.melSpectrogramModel = melSpectrogramModel
        self.babyCryDetectionModel = babyCryDetectionModel
        self.viewController = viewController
        self.inputNode = audioEngine.inputNode
        self.sampleRate = 22050.0 // Adjusted to match given sample rate
        self.cryTranslationLength = cryTranslationLength
        self.testFrequency = testFrequency
        self.detectionSensitivity = detectionSensitivity
        self.cryEndThreshold = cryEndThreshold
        self.bufferSize = AVAudioFrameCount(sampleRate * cryTranslationLength)
        self.bufferQueue = CircularBuffer<Float>(capacity: Int(sampleRate * cryTranslationLength))
        self.secondaryBufferQueue = CircularBuffer<Float>(capacity: Int(sampleRate * cryTranslationLength))
        
        setupAudioSession()
        setupAudioEngine()
    }
    
    private func setupAudioSession() {
        let audioSession = AVAudioSession.sharedInstance()
        try? audioSession.setCategory(.playAndRecord, mode: .default, options: [])
        try? audioSession.setPreferredSampleRate(sampleRate)
        try? audioSession.setActive(true)
    }
    
    private func setupAudioEngine() {
        let inputFormat = inputNode.inputFormat(forBus: 0)
        sleep(5)
        inputNode.installTap(onBus: 0, bufferSize: bufferSize, format: inputFormat) { [weak self] (buffer, time) in
            self?.processAudioBuffer(buffer: buffer)
        }
        
        audioEngine.prepare()
    }
    
    func startDetection() {
        try? audioEngine.start()
        startProcessingTimer()
    }
    
    func stopDetection() {
        processingTimer?.invalidate()
        processingTimer = nil
        audioEngine.stop()
        inputNode.removeTap(onBus: 0)
        
        // Clear or reset buffers to stop processing
        self.clearBuffers()
            
            // Reset UI or update labels as needed
        viewController?.updateStatusLabel(with: "Detection stopped")
    }
    
    // This function will make sure that it is called every 0.1 sec i.e processing interval
    private func startProcessingTimer() {
        processingTimer = Timer.scheduledTimer(withTimeInterval: testFrequency, repeats: true) { [weak self] timer in
            self?.runModelsOnBuffer()
        }
    }
    
     func stopProcessingTimer() {
        self.audioEngine.stop()
        processingTimer?.invalidate()
        processingTimer = nil
    }
    
    private func processAudioBuffer(buffer: AVAudioPCMBuffer) {
        guard let channelData = buffer.floatChannelData?[0] else { return }
        let frameLength = Int(buffer.frameLength)
        
        for i in 0..<frameLength {
            bufferQueue.push(channelData[i])
            secondaryBufferQueue.push(channelData[i])
        }
        
        //This will maintain a buffer size of 2 seconds of latest audio chunck
        while bufferQueue.count > Int(bufferSize) {
            _ = bufferQueue.pop()
        }
        
        // Maintain the additional buffer size of 1.2 seconds
        while secondaryBufferQueue.count > Int(bufferSize) {
            _ = secondaryBufferQueue.pop()
        }
    }
    
    func getDocumentsDirectory() -> URL {
        let paths = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
        print("Audio Path: ", paths[0])
        return paths[0]
    }
    
    private func runModelsOnBuffer() {
        if self.viewController?.isDetecting == true {
            let audioData = bufferQueue.toArray()
            guard let melSpectrogram = createMelSpectrogram(from: audioData) else { return }
            guard let babyCryDetected = detectBabyCry(using: melSpectrogram) else { return }
            
            //send audio if baby cry reaches 18 i.e. 1.8 seconds
            if babyCryDetected {
                
                self.viewController?.cryStatusLabel.text = "Baby Cry Started"
                if detectInRow == 0 {
                    detectionStartIndex = bufferQueue.count - Int(sampleRate * cryTranslationLength)
                }
                detectInRow += 1
                noCryInRow = 0 // reset noCryInRow since baby cry is detected
                print("Detection in a row:", self.detectInRow)
                self.viewController?.cryOutputLabel.text = "Baby Cry in a row:\(self.detectInRow)"
                if detectInRow == detectionSensitivity {
                    triggerAlert()
                    detectInRow = 0
                    detectionStartIndex = nil
                }
            } else {
                self.viewController?.cryStatusLabel.text = "Baby Cry Stopped"
                detectInRow = 0
                detectionStartIndex = nil
                noCryInRow += 1
                print("Non cry in a row:", self.noCryInRow)
                self.viewController?.cryOutputLabel.text = "Baby Non-Cry in row:\(self.noCryInRow)"
                if noCryInRow == cryEndThreshold {
                    viewController?.updateStatusLabel(with: "Baby cry ended!")
                    noCryInRow = 0
                }
            }
        }
    }
    
    func clearBuffers() {
        bufferQueue.clear()
        secondaryBufferQueue.clear()
    }
    
    private func createMelSpectrogram(from audioData: [Float]) -> MLMultiArray? {
        let audioBuffer = try? MLMultiArray(shape: [1, 44100], dataType: .float32) // Adjusted to match model input shape

        for (index, value) in audioData.enumerated() {
            audioBuffer?[index] = NSNumber(value: value)
        }
        do {
            let model = try wave_to_logmel(configuration: MLModelConfiguration())
            let input = wave_to_logmelInput(x_1: audioBuffer!)
            let prediction = try model.prediction(input: input)
            let output = prediction.var_62
            return output
        } catch {
            print("Failed to generate mel spectrogram: \(error.localizedDescription)")
            return nil
        }
    }
    
    private func detectBabyCry(using melSpectrogram: MLMultiArray) -> Bool? {
        do {
            let model = try encodedBabyCryDetectionModel(configuration: MLModelConfiguration())
            let input = encodedBabyCryDetectionModelInput(audio: melSpectrogram)
            let prediction = try model.prediction(input: input)
            let logitArray = prediction.var_104ShapedArray
            let logitSlice = logitArray[0]
            guard let logit = logitSlice.scalar else {
                print("Failed to extract logit value.")
                return nil
            }
            let sigmoid = 1 / (1 + exp(-logit))
            let threshold: Float = 0.6
            let babyCryDetected = sigmoid >= threshold
            
            print("Raw Output:\(logitArray), Sigmoid:\(sigmoid), Baby Cry:\(babyCryDetected)")
            return babyCryDetected
        } catch {
            print("Failed to generate mel spectrogram: \(error.localizedDescription)")
            return nil
        }
    }
    
    private func triggerAlert() {
        viewController?.updateStatusLabel(with: "Baby cry detected!")
        if let detectionStartIndex = detectionStartIndex, detectionStartIndex >= 0 {
            let additionalAudioChunk = secondaryBufferQueue.toArray()
            if detectionStartIndex < additionalAudioChunk.count {
                let completeAudioChunk = additionalAudioChunk[detectionStartIndex..<additionalAudioChunk.count]
                viewController?.handleBabyCryDetected(audioChunk: Array(completeAudioChunk))
            }
        }
    }
}

class CircularBuffer<T> {
    private var buffer: [T?]
    private var readIndex = 0
    private var writeIndex = 0
    private var size = 0
    
    var count: Int {
        return size
    }
    
    var capacity: Int {
        return buffer.count
    }
    
    init(capacity: Int) {
        self.buffer = [T?](repeating: nil, count: capacity)
    }
    
    func push(_ element: T) {
        buffer[writeIndex] = element
        writeIndex = (writeIndex + 1) % buffer.count
        
        if size < buffer.count {
            size += 1
        } else {
            readIndex = (readIndex + 1) % buffer.count
        }
    }
    
    func pop() -> T? {
        guard size > 0 else { return nil }
        
        let element = buffer[readIndex]
        buffer[readIndex] = nil
        readIndex = (readIndex + 1) % buffer.count
        size -= 1
        
        return element
    }
    
    func clear() {
        buffer = [T?](repeating: nil, count: buffer.count)
        readIndex = 0
        writeIndex = 0
        size = 0
    }
    
    func toArray() -> [T] {
        var array = [T]()
        for i in 0..<size {
            let index = (readIndex + i) % buffer.count
            if let element = buffer[index] {
                array.append(element)
            }
        }
        return array
    }
}

extension CircularBuffer where T: ExpressibleByFloatLiteral {
    func toFloatArray() -> [Float] {
        return toArray().compactMap { $0 as? Float }
    }
}
