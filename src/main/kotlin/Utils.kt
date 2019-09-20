import org.apache.commons.math3.linear.MatrixUtils
import org.apache.commons.math3.linear.RealMatrix
import java.awt.image.BufferedImage
import java.io.File
import java.nio.file.Files
import java.nio.file.Paths
import java.util.Locale
import javax.imageio.ImageIO
import kotlin.collections.ArrayList
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
            m.setEntry(row, column, rnd.nextDouble() - 0.5)
    return m
}

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
            result.setEntry(row, column, x - getEntry(row, column))
    return result
}


fun mnistCsvToJpg(inputFilePath: String, outputFilePath: String, index: Int) {
    val image = BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY)
    val lines = Files.readAllLines(Paths.get(inputFilePath))
    var pixels = lines[index].split(",")
    pixels = pixels.subList(1, pixels.size)
    var column: Int
    var row = -1

    for (i in 0 until pixels.size) {
        column = i % 28
        if (column == 0) {
            row++
        }
        var colorValue = pixels[i].toInt()
        if (colorValue == 0) {
            colorValue = Int.MAX_VALUE
        }
        image.setRGB(column, row, colorValue)
    }

    ImageIO.write(image, "jpg", File(outputFilePath))
}

fun RealMatrix.store(outputFilePath: String) {
    val fpFormat = "%.8f"
    val separator = " "
    val s = StringBuilder(rowDimension * columnDimension * 2)
    for (row in 0 until rowDimension) {
        for (column in 0 until columnDimension-1) {
            val value= getEntry(row, column)
            if (value >= 0) {
                s.append(separator)
            }
            s.append(separator)
            s.append(String.format(Locale.ROOT, fpFormat, value))
        }
        val value= getEntry(row, columnDimension-1)
        if (value >= 0) {
            s.append(separator)
        }
        s.append(separator)
        s.appendln(String.format(Locale.ROOT, fpFormat, value))

    }
    File(outputFilePath).printWriter().use { it.println(s)}
}

fun RealMatrix.load(inputFilePath: String) {
    val separator = " "
    val lines = Files.readAllLines(Paths.get(inputFilePath))
    lines.forEachIndexed { rowIndex, line ->
        val row = line.split(separator)
        row.filter{it.isNotEmpty()}.forEachIndexed {columnIndex, columnValue ->
            setEntry(rowIndex, columnIndex, columnValue.replace(",", ".").toDouble())
        }
    }
}


fun scaledInputsFromCsvString(csvStr: String): ArrayList<Double> {
    val scaledList = ArrayList<Double>()
    var pixels = csvStr.split(",")
    for (i in 1 until pixels.size) {
        scaledList.add( (pixels[i].toDouble() / 255.0 * 0.99) + 0.1)
    }
    return scaledList
}


fun scaledTargetsFromCsvString(nOutputs: Int, targetDigit: Int): DoubleArray {
    val tempList = List(nOutputs) {0.01}
    val targets = tempList.toDoubleArray()
    targets[targetDigit] = 0.99
    return targets
}

fun targetDigit(csvStr: String): String {
    var pixels = csvStr.split(",")
    return pixels[0]
}

fun main() {
    /*
    val inputFile = "resources/mnist_train_100.csv"
    val outputFile = "resources/mnist_train_100.jpg"
    val index = 1
    mnistCsvToJpg(inputFile, outputFile, index)
    */


    var matrix = createRandom(5, 5)
    matrix.store("j:\\f.txt")


    matrix = MatrixUtils.createRealMatrix(5, 5)
    matrix.load("j:\\f.txt")
}
