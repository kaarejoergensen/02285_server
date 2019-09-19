package domain.gridworld.hospital.ui_components;

/*
    Fredrik Kloster
    Help class for describing the current values for the canvas

 */

import domain.gridworld.hospital.HospitalDomain;

public class CanvasDetails {
    public static final double BOX_MARGIN_PERCENT = 0.1;
    public static final double TEXT_MARGIN_PERCENT = 0.2;

    public int originLeft, originTop;
    public int width, height;
    public int cellSize, cellBoxMargin, cellTextMargin;

    public CanvasDetails(){
    }


    public void recalculateCanvas(int bufferWidth, int numCols, int bufferHeight, int numRows){
        cellSize = Math.min(bufferWidth / numCols, bufferHeight / numRows);

        int excessWidth = bufferWidth - numCols * cellSize;
        int excessHeight = bufferHeight - numRows * cellSize;

        originLeft = excessWidth / 2;
        originTop = excessHeight / 2;
        width = bufferWidth - excessWidth;
        height = bufferHeight - excessHeight;

        cellBoxMargin = (int) (this.cellSize * BOX_MARGIN_PERCENT);
        cellTextMargin = (int) (this.cellSize * TEXT_MARGIN_PERCENT);
    }

}
