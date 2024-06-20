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
    private var babyCryDetector: BabyCryDetector?
    private let statusLabel = UILabel()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
        do {
            let melSpectrogramModel = try wave_to_logmel(configuration: MLModelConfiguration())
            let babyCryDetectionModel = try encodedBabyCryDetectionModel(configuration: MLModelConfiguration())
            babyCryDetector = BabyCryDetector(melSpectrogramModel: melSpectrogramModel.model, babyCryDetectionModel: babyCryDetectionModel.model, viewController: self)
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
    
    func sigmoid(_ x: Double) -> Double {
        return 1 / (1 + exp(-x))
    }

    func roundSigmoid(_ out: Double) -> Int {
        let sigmoidValue = sigmoid(out)
        let roundedValue = round(sigmoidValue)
        return Int(roundedValue)
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
    private let bufferLengthInSeconds: Double = 1.0
    private let processingInterval: Double = 0.1
    private weak var viewController: ViewController?
    private var detectInRow: Int = 0
    private let sensitivity: Int = 3
    
    init(melSpectrogramModel: MLModel, babyCryDetectionModel: MLModel, viewController: ViewController) {
        self.melSpectrogramModel = melSpectrogramModel
        self.babyCryDetectionModel = babyCryDetectionModel
        self.viewController = viewController
        self.inputNode = audioEngine.inputNode
        self.sampleRate = 44100.0
        self.bufferSize = AVAudioFrameCount(sampleRate * bufferLengthInSeconds)
        self.bufferQueue = CircularBuffer<Float>(capacity: Int(sampleRate))
        
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
        
        inputNode.installTap(onBus: 0, bufferSize: bufferSize, format: inputFormat) { [weak self] (buffer, time) in
            self?.processAudioBuffer(buffer: buffer)
        }
        
        audioEngine.prepare()
    }
    
    func startDetection() {
        try? audioEngine.start()
    }
    
    private func processAudioBuffer(buffer: AVAudioPCMBuffer) {
        guard let channelData = buffer.floatChannelData?[0] else { return }
        let frameLength = Int(buffer.frameLength)
        
        for i in 0..<frameLength {
            bufferQueue.push(channelData[i])
        }
        
        if bufferQueue.count >= Int(bufferSize) {
            runModelsOnBuffer()
        }
    }
    
    private func runModelsOnBuffer() {
        let audioData = bufferQueue.toArray()//Array(bufferQueue)
        guard let melSpectrogram = createMelSpectrogram(from: audioData) else { return }
        guard let babyCryDetected = detectBabyCry(using: melSpectrogram) else { return }
        
        if babyCryDetected {
            triggerAlert()
        }
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
              //Assuming the output property is a float value indicating the logit
              let logitArray = prediction.var_104ShapedArray
              // Extract the logit value from the MLShapedArray
              let logitSlice = logitArray[0]
              guard let logit = logitSlice.scalar else {
                  print("Failed to extract logit value.")
                  return nil
              }
  //            print("Logit value:", logit)
              let sigmoid = 1 / (1 + exp(-logit))
  //            print("Probability:", probability)
              
              // Use a threshold of 0.5 to convert rawValue to 0 or 1
                  let threshold: Float = 0.5
                  let babyCryDetected = sigmoid >= threshold
              
              print("Raw Output:\(logitArray), Sigmoid:\(sigmoid), Baby Cry:\(babyCryDetected)")
              
              //Check for baby cry loop for 18 consecutive times
              if babyCryDetected == true { // Change the condition to check for noise
                  self.detectInRow += 1
                  print("Detection in a row:", self.detectInRow)
                  if self.detectInRow == self.sensitivity {
                      self.detectInRow = 0
                      return true
                  } else {
                      return false
                  }
              } else {
                  self.detectInRow = 0 // Reset counter if no noise is detected
                  return false
              }
              } catch {
                  print("Failed to generate mel spectrogram: \(error.localizedDescription)")
                  return nil
              }
      }
    
    private func triggerAlert() {
        viewController?.updateStatusLabel(with: "Baby cry detected!")
        // Additional alert mechanisms can be implemented here.
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






//class ViewController: UIViewController {
//
//    var recordingSession: AVAudioSession!
//    var audioRecorder: AVAudioRecorder!
//    var fileName: URL = URL(fileURLWithPath: "")
//    var stopRecordingBool = false
//
//    override func viewDidLoad() {
//        super.viewDidLoad()
//        // Do any additional setup after loading the view.
//    }
//
//    func recordingSessionStart() {
//        recordingSession = AVAudioSession.sharedInstance()
//            do {
//                try recordingSession.setCategory(AVAudioSession.Category.playAndRecord, mode: .videoRecording)
//                try recordingSession.setActive(true)
//                recordingSession.requestRecordPermission() { [unowned self] allowed in
//                    DispatchQueue.main.async {
//                        if allowed {
//                            self.startRecording()
////                            self.startTimer()
//                        } else {
//                            // failed to record!
//                        }
//                    }
//                }
//            } catch {
//                // failed to record!
//            }
//    }
//
//    func startRecording() {
//        fileName = getDocumentsDirectory().appendingPathComponent("recording.wav")
//
//        let settings = [
//            AVFormatIDKey: Int(kAudioFormatMPEG4AAC),
//            AVSampleRateKey: 44100,
//            AVNumberOfChannelsKey: 1,
//            AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue
//        ] as [String : Any]
//                do{
//                    audioRecorder = try AVAudioRecorder(url: fileName, settings: settings)
//                    if(audioRecorder.record(forDuration: 2 )) {
//                        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
//                            print("recording succesfull...")
//                            if self.audioRecorder != nil {
//                                self.audioRecorder.stop()
//                                self.audioRecorder = nil
//                                if self.stopRecordingBool == false {
////                                    self.sendAudioToInference()
//                                    //convert the audio mel and get the result here
//                                    guard let audioFile = try? AVAudioFile(forReading: self.fileName) else {
//                                        print("Failed to initialize AVAudioFile.")
//                                    }
//                                    // Get the format of the audio file
//                                    let format = audioFile.processingFormat
//                                    // Get the length of the audio file in frames
//                                    let frameCount = AVAudioFrameCount(audioFile.length)
//                                    // Create a buffer to hold the audio data
//                                    guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
//                                        print("Failed to create PCM buffer.")
//                                    }
//                                    // Read the audio file into the buffer
//                                    guard (try? audioFile.read(into: buffer)) != nil else {
//                                        print("Failed to read audio file into buffer.")
//                                    }
//                                }
//                            }
//                        }
//                    }
//                }
//                catch {
//                    print ("failed...")
//                }
//    }
//
//    func getDocumentsDirectory() -> URL {
//        let paths = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
//        print("Audio Path: ", paths[0])
//        return paths[0]
//    }
//
//    func generateMelSpectrogram() -> MLMultiArray? {
////        let monoSignal: [Float] =
////        print("Mono Signal:", monoSignal.prefix(upTo: 5))
//
//        guard let tensor = try? MLMultiArray(shape: [1, NSNumber(value: monoSignal.count)], dataType: .float32) else {
//            print("Failed to create MLMultiArray.")
//            return nil
//        }
//
//        var monoSignal = [Float](repeating: 0, count: Int(audioData.frameLength))
//        print("Mono Signal:", monoSignal.prefix(upTo: 5))
//
//        for i in 0..<monoSignal.count {
//            tensor[i] = NSNumber(value: monoSignal[i])
//        }
//        print("Tensor Shape:", tensor.shape)
//        print("Tensor DataType:", tensor.dataType)
//        do {
//            let model = try wave_to_logmel(configuration: MLModelConfiguration())
//            let input = wave_to_logmelInput(x_1: tensor)
//            let prediction = try model.prediction(input: input)
//            print("Model input created:", input)
//            print("Feature Names:", prediction.featureNames)
//            print("var_62:", prediction.var_62)
//            print("var_62 Shaped Array:", prediction.var_62ShapedArray)
//            print("var_62 Values:", prediction.featureValue(for: "var_62")!)
//
//            let model1 = try encodedBabyCryDetectionModel(configuration: MLModelConfiguration())
//            let input1 = encodedBabyCryDetectionModelInput(audio: prediction.var_62ShapedArray)
//            let prediction1 = try model1.prediction(input: input1)
//            print("Model1 input created:", input1)
//            print("Feature Names1:", prediction1.featureNames)
//            print("var_104:", prediction1.var_104)
//            print("var_104 Shaped Array:", prediction1.var_104ShapedArray)
//            print("var_104 Values:", prediction1.featureValue(for: "var_104")!)
//
//            //Assuming the output property is a float value indicating the logit
//            let logitArray = prediction1.var_104ShapedArray // Replace 'output' with the actual property name
//            print("Logit array:", logitArray)
//            // Extract the logit value from the MLShapedArray
//            let logitSlice = logitArray[0]
//            guard let logit = logitSlice.scalar else {
//                print("Failed to extract logit value.")
//                return nil
//            }
//            print("Logit value:", logit)
//            let probability = 1 / (1 + exp(-logit))
//            print("Probability:", probability)
//            return tensor
//        } catch {
//            print("Failed to generate mel spectrogram: \(error.localizedDescription)")
//            return nil
//        }
//    }
//}

