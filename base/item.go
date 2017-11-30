package base

import (
	"c4science.ch/source/gokick/kern"
	"c4science.ch/source/gokick/score"
)

type Item struct {
	score.Process
}

func NewItem(kernel kern.Kernel) *Item {
	return &Item{
		Process: *score.NewProcess(kernel),
	}
}
