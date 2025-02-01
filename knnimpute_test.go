package knnimpute

import (
	"math"
	"testing"

)

func TestKNNImpute(t *testing.T) {
	data := [][]float64{
		{1.0, 2.0, math.NaN()},
		{2.0, 3.0, 4.0},
		{3.0, 4.0, 5.0},
	}

	expected := [][]float64{
		{1.0, 2.0, 4.5}, // Imputed value should be the mean of 4.0 and 5.0
		{2.0, 3.0, 4.0},
		{3.0, 4.0, 5.0},
	}

	result := KNNImpute(data, 2)

	if math.Abs(result[0][2]-expected[0][2]) > 1e-6 {
		t.Errorf("Expected %v, but got %v", expected, result)
	}
}
