import org.apache.commons.math3.linear.MatrixUtils
import org.apache.commons.math3.linear.RealMatrix
import kotlin.math.exp
import kotlin.random.Random

fun sigm(x: Double) = 1 / (1 + exp(-x))

fun RealMatrix.sigm(): RealMatrix {
    val z: Array<DoubleArray?> = arrayOfNulls(this.data.size)
    var i = 0
    data.iterator().forEach {
        z[i++] = doubleArrayOf(sigm(it[0]))
    }
    return MatrixUtils.createRealMatrix(z)
}

fun createRandom(nRows: Int, nColumns: Int): RealMatrix {
    val rnd = Random
    val m = MatrixUtils.createRealMatrix(nRows, nColumns)
    for (row in 0 until nRows)
        for (column in 0 until nColumns)
            m.setEntry(row, column, rnd.nextDouble()-0.5)
    return m
}

//fun createColumn(vararg column: Double) = MatrixUtils.createColumnRealMatrix(column)
fun createColumn(column: DoubleArray) = MatrixUtils.createColumnRealMatrix(column)

fun RealMatrix.scalarMultiple(other: RealMatrix): RealMatrix? {
    if (columnDimension != other.columnDimension || rowDimension != other.rowDimension) {
        throw IllegalArgumentException("Dimensions error")
    } else {
        val result = MatrixUtils.createRealMatrix(rowDimension, columnDimension)
        for (row in 0 until rowDimension)
            for (column in 0 until columnDimension)
                result.setEntry(row, column,
                        getEntry(row, column) * other.getEntry(row, column))
        return result
    }
}

fun RealMatrix.scalarSubtractionFrom(x: Double): RealMatrix {
    val result = MatrixUtils.createRealMatrix(rowDimension, columnDimension)
    for (row in 0 until rowDimension)
        for (column in 0 until columnDimension)
            result.setEntry(row, column,
                    x - getEntry(row, column))
    return result
}
