import org.apache.commons.math3.linear.RealMatrix

class NeuralNetwork (nInput: Int, nHidden: Int, nOutput: Int,
                     val learningRate: Int) {
    private val weightInputHidden = createRandom(nHidden, nInput)
    private val weightHiddenOutput = createRandom(nOutput, nHidden)


    fun query(vararg inputValues: Double): RealMatrix {
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


    }
}

fun main() {
    val nInput = 3
    val nHidden = 3
    val nOutput = 3

    val lRate = 2

    val net = NeuralNetwork(nInput, nHidden, nOutput, lRate)

    val res = net.query(1.0, 2.0, 3.0)
    print(res)
}