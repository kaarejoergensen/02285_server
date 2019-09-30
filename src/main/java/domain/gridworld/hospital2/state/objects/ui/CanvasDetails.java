package domain.gridworld.hospital2.state.objects.ui;

import lombok.Getter;
import lombok.NoArgsConstructor;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.javatuples.Pair;

import java.awt.*;
import java.awt.font.FontRenderContext;
import java.awt.font.TextLayout;

@NoArgsConstructor
public class CanvasDetails {
    private static Logger serverLogger = LogManager.getLogger("server");

    private static final double BOX_MARGIN_PERCENT = 0.1;
    private static final double TEXT_MARGIN_PERCENT = 0.2;

    @Getter private int originLeft, originTop;
    @Getter private int width, height;
    @Getter private int cellSize, cellBoxMargin, cellTextMargin;
    @Getter private Pair<Integer, Integer> moveArm, pushArm;

    @Getter Font currentFont;
    @Getter FontRenderContext fontRenderContext;

    public void calculate(Graphics2D g, int bufferWidth, int numCols, int bufferHeight, int numRows) {
        this.calculateCanvas(bufferWidth, numCols, bufferHeight, numRows);
        this.calculateFont(g);
    }

    private void calculateCanvas(int bufferWidth, int numCols, int bufferHeight, int numRows){
        cellSize = Math.min(bufferWidth / numCols, bufferHeight / numRows);

        int excessWidth = bufferWidth - numCols * cellSize;
        int excessHeight = bufferHeight - numRows * cellSize;

        originLeft = excessWidth / 2;
        originTop = excessHeight / 2;
        width = bufferWidth - excessWidth;
        height = bufferHeight - excessHeight;

        cellBoxMargin = (int) (this.cellSize * BOX_MARGIN_PERCENT);
        cellTextMargin = (int) (this.cellSize * TEXT_MARGIN_PERCENT);

        this.moveArm = Pair.with(this.cellSize / 2 -1, (int) (this.cellSize * 0.60));
        this.pushArm = Pair.with((int) (this.cellSize / Math.sqrt(2.0)),
                (int) ((this.cellSize - 2 * this.cellBoxMargin) / Math.sqrt(2.0)));
    }

    private void calculateFont(Graphics2D g) {
        this.fontRenderContext = g.getFontRenderContext();
        int fontSize = 0;
        Font nextFont = new Font(null, Font.BOLD, fontSize);
        Rectangle bounds;
        do {
            this.currentFont = nextFont;
            fontSize++;
            nextFont = new Font(null, Font.BOLD, fontSize);
            // FIXME: Holy shit, creating a TextLayout object is SLOW!
            long t1 = System.nanoTime();
            var text = new TextLayout("W", nextFont, fontRenderContext); // Using W because it's wide.
            serverLogger.debug(String.format("fontSize: %d us.", (System.nanoTime() - t1) / 1000));
            bounds = text.getPixelBounds(fontRenderContext, 0, 0);
        } while (bounds.width < this.cellSize - 2 * this.cellTextMargin &&
                bounds.height < this.cellSize - 2 * this.cellTextMargin);
    }
}
