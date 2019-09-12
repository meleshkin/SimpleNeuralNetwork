import org.apache.commons.math3.linear.MatrixUtils
import org.apache.commons.math3.linear.RealMatrix
import kotlin.math.exp
import kotlin.random.Random

fun main() {
    val m = createRandom(2, 3)
    print(m)
    val m1 = createColumn(doubleArrayOf(1.0, 2.0, 3.0))
    val m2 = m.multiply(m1)
    print(m2)
}

fun main1() {


    val INPUT_DATA = arrayOf(
            doubleArrayOf(0.9),
            doubleArrayOf(0.1),
            doubleArrayOf(0.8)
    )
    val INPUT = MatrixUtils.createRealMatrix(INPUT_DATA)

    //MatrixUtils.createRe

    val W_INPUT_HIDDEN_DATA = arrayOf(
            doubleArrayOf(0.9, 0.3, 0.4),
            doubleArrayOf(0.2, 0.8, 0.2),
            doubleArrayOf(0.1, 0.5, 0.6)
    )
    val W_INPUT_HIDDEN = MatrixUtils.createRealMatrix(W_INPUT_HIDDEN_DATA)

    val X_HIDDEN = W_INPUT_HIDDEN.multiply(INPUT)

    val O_HIDDEN = X_HIDDEN.sigm()

    val W_HIDDEN_OUTPUT_DATA = arrayOf(
            doubleArrayOf(0.3, 0.7, 0.5),
            doubleArrayOf(0.6, 0.5, 0.2),
            doubleArrayOf(0.8, 0.1, 0.9)
    )

    val W_HIDDEN_OUTPUT = MatrixUtils.createRealMatrix(W_HIDDEN_OUTPUT_DATA)

    val X_OUTPUT = W_HIDDEN_OUTPUT.multiply(O_HIDDEN)

    val O_OUTPUT = X_OUTPUT.sigm()

    print(O_OUTPUT)


}