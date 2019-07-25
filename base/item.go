package base

import (
	"github.com/lucasmaystre/gokick/kern"
	"github.com/lucasmaystre/gokick/score"
)

type Item struct {
	score.Process
}

func NewItem(kernel kern.Kernel) *Item {
	return &Item{
		Process: *score.NewProcess(kernel),
	}
}
