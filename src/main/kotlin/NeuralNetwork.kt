import org.apache.commons.math3.linear.RealMatrix

class NeuralNetwork (nInput: Int, nHidden: Int, nOutput: Int,
                     private val learningRate: Double) {
    private var weightInputHidden = createRandom(nHidden, nInput)
    private var weightHiddenOutput = createRandom(nOutput, nHidden)


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

    val nInput = 3
    val nHidden = 3
    val nOutput = 3
    val lRate = 0.3

    val net = NeuralNetwork(nInput, nHidden, nOutput, lRate)

    net.train(doubleArrayOf(1.0, 2.0, 3.0), doubleArrayOf(1.0, 2.0, 3.0))


}
