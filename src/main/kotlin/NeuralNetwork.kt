import org.apache.commons.math3.linear.RealMatrix
import java.nio.file.Files
import java.nio.file.Paths

class NeuralNetwork (nInput: Int, nHidden: Int, nOutput: Int,
                     private val learningRate: Double) {
    private var weightInputHidden = createRandom(nHidden, nInput)
    private var weightHiddenOutput = createRandom(nOutput, nHidden)


    fun query(inputValues: DoubleArray): RealMatrix {
        val input = createColumn(inputValues)

        val hiddenInputs = weightInputHidden.multiply(input)
        val hiddenOutputs = hiddenInputs.sigm()

        val finalInputs = weightHiddenOutput.multiply(hiddenOutputs)
        val finalOutputs = finalInputs.sigm()

        return finalOutputs
    }

    fun train(inputValues: DoubleArray, targetValues: DoubleArray) {
        val input = createColumn(inputValues)
        val target = createColumn(targetValues)

        val hiddenInputs = weightInputHidden.multiply(input)
        val hiddenOutputs = hiddenInputs.sigm()

        val finalInputs = weightHiddenOutput.multiply(hiddenOutputs)
        val finalOutputs = finalInputs.sigm()

        val outputError = target.subtract(finalOutputs)
        val hiddenError = weightHiddenOutput.transpose().multiply(outputError)

        val weightHiddenOutputDelta =
                outputError.scalarMultiple(finalOutputs)!!
                    .scalarMultiple(finalOutputs.scalarSubtractionFrom(1.0))!!
                    .multiply(hiddenOutputs.transpose())!!
                    .scalarMultiply(learningRate)

        weightHiddenOutput = weightHiddenOutput.add(weightHiddenOutputDelta)

        val weightInputHiddenDelta =
                hiddenError.scalarMultiple(hiddenOutputs)!!
                    .scalarMultiple(hiddenOutputs.scalarSubtractionFrom(1.0))!!
                    .multiply(input.transpose())!!
                    .scalarMultiply(learningRate)

        weightInputHidden = weightInputHidden.add(weightInputHiddenDelta)

    }
}

fun main() {
    val nInput = 784
    val nHidden = 100
    val nOutput = 10
    val lRate = 0.3

    val net = NeuralNetwork(nInput, nHidden, nOutput, lRate)

    val lines = Files.readAllLines(Paths.get("resources/mnist_train_100.csv"))
    lines.forEach {

        val digit = targetDigit(it).toInt()
        val targets = scaledTargetsFromCsvString(nOutput, digit)
        val inputs = createColumn(scaledInputsFromCsvString(it).toDoubleArray())

        net.train(inputs.getColumn(0), targets)
    }

    val testString = "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,11,150,253,202,31,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,37,251,251,253,107,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,21,197,251,251,253,107,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,110,190,251,251,251,253,169,109,62,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,253,251,251,251,251,253,251,251,220,51,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,182,255,253,253,253,253,234,222,253,253,253,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,63,221,253,251,251,251,147,77,62,128,251,251,105,0,0,0,0,0,0,0,0,0,0,0,0,0,0,32,231,251,253,251,220,137,10,0,0,31,230,251,243,113,5,0,0,0,0,0,0,0,0,0,0,0,0,37,251,251,253,188,20,0,0,0,0,0,109,251,253,251,35,0,0,0,0,0,0,0,0,0,0,0,0,37,251,251,201,30,0,0,0,0,0,0,31,200,253,251,35,0,0,0,0,0,0,0,0,0,0,0,0,37,253,253,0,0,0,0,0,0,0,0,32,202,255,253,164,0,0,0,0,0,0,0,0,0,0,0,0,140,251,251,0,0,0,0,0,0,0,0,109,251,253,251,35,0,0,0,0,0,0,0,0,0,0,0,0,217,251,251,0,0,0,0,0,0,21,63,231,251,253,230,30,0,0,0,0,0,0,0,0,0,0,0,0,217,251,251,0,0,0,0,0,0,144,251,251,251,221,61,0,0,0,0,0,0,0,0,0,0,0,0,0,217,251,251,0,0,0,0,0,182,221,251,251,251,180,0,0,0,0,0,0,0,0,0,0,0,0,0,0,218,253,253,73,73,228,253,253,255,253,253,253,253,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,113,251,251,253,251,251,251,251,253,251,251,251,147,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,31,230,251,253,251,251,251,251,253,230,189,35,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,62,142,253,251,251,251,251,253,107,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,72,174,251,173,71,72,30,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"
    val input = createColumn(scaledInputsFromCsvString(testString).toDoubleArray()).getColumn(0)
    val resultArray = net.query(input)
    val result = resultArray.getColumn(0).indexOf(resultArray.getColumn(0).max()!!)
    print(result)
}
