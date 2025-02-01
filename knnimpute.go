package knnimpute

import (
	"math"
	"sort"
)

type Neighbour struct {
	Index    int
	Distance float64
}

// KNNImpute performs k-nearest neighbour imputation on a 2D slice of float64
func KNNImpute(data [][]float64, k int) [][]float64 {
	rows := len(data)
	cols := len(data[0])
	imputedData := make([][]float64, rows)
	for i := range imputedData {
		imputedData[i] = make([]float64, cols)
		copy(imputedData[i], data[i])
	}

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if math.IsNaN(data[i][j]) {
				neighbours := findKNearestNeighbours(data, i, j, k)
				if len(neighbours) > 0 {
					imputedData[i][j] = mean(neighbours)
				}
			}
		}
	}
	return imputedData
}

func findKNearestNeighbours(data [][]float64, row, column, k int) []float64 {
	distances := []Neighbour{}
	for i := 0; i < len(data); i++ {
		if i == row {
			continue
		}
		distance := euclideanDistance(data[row], data[i], column)
		if !math.IsNaN(distance) {
			distances = append(distances, Neighbour{i, distance})
		}
	}

	sort.Slice(distances, func(a, b int) bool {
		return distances[a].Distance < distances[b].Distance
	})

	neighbours := []float64{}
	count := 0
	for _, neighbour := range distances {
		if count >= k {
			break
		}
		if !math.IsNaN(data[neighbour.Index][column]) {
			neighbours = append(neighbours, data[neighbour.Index][column])
			count++
		}
	}

	return neighbours
}

func euclideanDistance(row1, row2 []float64, ignoreColumn int) float64 {
	sum := 0.0
	count := 0
	for i := range row1 {
		if i == ignoreColumn || math.IsNaN(row1[i]) || math.IsNaN(row2[i]) {
			continue
		}
		diff := row1[i] - row2[i]
		sum += diff * diff
		count++
	}
	if count == 0 {
		return math.NaN()
	}
	return math.Sqrt(sum)
}

func mean(values []float64) float64 {
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}
