package gui.widgets;

import domain.Domain;
import server.Server;

import javax.swing.JPanel;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.image.VolatileImage;

import static java.awt.RenderingHints.KEY_ANTIALIASING;
import static java.awt.RenderingHints.KEY_TEXT_ANTIALIASING;
import static java.awt.RenderingHints.VALUE_ANTIALIAS_ON;
import static java.awt.RenderingHints.VALUE_TEXT_ANTIALIAS_ON;

public class DomainPanel
        extends JPanel
{
    /**
     * TODO: Benchmark this against BufferedImage version when domain actually does some real rendering.
     */
    private VolatileImage domainBackgroundBuffer;
    private VolatileImage stateBackgroundBuffer;
    private VolatileImage stateTransitionBuffer;
    private Graphics2D domainBackgroundGraphics;
    private Graphics2D stateBackgroundGraphics;
    private Graphics2D stateTransitionGraphics;

    private Domain domain;
    private Thread domainRenderingThread;
    private static int renderingThreadCount = 0;

    private double lastStateInterpolation = 0;
    private double currentStateInterpolation;
    private boolean requireFullRender = false;
    private boolean isNewlyRendered = false;

    private boolean shutdown = false;
    private boolean signal = false;

    public DomainPanel(Domain domain)
    {
        super();
        this.setOpaque(true);

        this.domain = domain;

        /*
         * Thread which renders the buffers off the EDT. The rendering begins on the signal, and the EDT
         * can wait for rendering to finish before painting the buffers to screen if newly rendered.
         */
        this.domainRenderingThread = new Thread(this::renderLoop,
                                                "DomainRenderingThread-" + DomainPanel.renderingThreadCount++);
    }

    @Override
    public void paint(Graphics g)
    {
        g.drawImage(this.stateTransitionBuffer, 0, 0, null);
    }

    /**
     * Repaints the buffers to the screen if the rendering thread rendered new content in its last rendering pass.
     * <p>
     * IMPORTANT: Must only be called by the EDT, after waiting on waitRenderFinish().
     */
    public void repaintIfNewlyRendered()
    {
        if (this.isNewlyRendered)
        {
            this.paintImmediately(0, 0, this.getWidth(), this.getHeight());
        }
    }

    public void startRenderingThread()
    {
        this.domainRenderingThread.start();
    }

    /**
     * Signals the rendering thread for this DomainPanel to shut down, and waits for it to join.
     */
    public void shutdownRenderingThread()
    {
        synchronized (this)
        {
            this.shutdown = true;
            this.signal = true;
            this.notifyAll();
        }
        while (true)
        {
            try
            {
                this.domainRenderingThread.join();
                return;
            }
            catch (InterruptedException ignored)
            {
            }
        }
    }

    /**
     * Signals the rendering thread for this DomainPanel to render the given state interpolation.
     */
    public synchronized void signalRenderBegin(double stateInterpolation)
    {
        this.validateBuffers();
        this.currentStateInterpolation = stateInterpolation;
        this.signal = true;
        this.notifyAll();
    }

    /**
     * Wait for this DomainPanel's rendering thread to finish rendering after the last call to signalRenderBegin().
     */
    public synchronized void waitRenderFinish()
    {
        while (this.signal)
        {
            try
            {
                this.wait();
            }
            catch (InterruptedException ignored)
            {
            }
        }
    }

    /**
     * Reallocates the buffers to match the current display device and size of the panel, if necessary.
     * <p>
     * FIXME: Sometimes when transitioning from windowed mode to fullscreen, it seems the change in size happens over
     * two steps, which causes us to reallocate twice. Why does the resize happen in two steps? Can we avoid it?
     * .
     * The debugging code is currently commented out, but if re-enabled it will show that two consecutive
     * calls to validateBuffers by the EDT will (sometimes) transition the window to fullscreen in two steps.
     * The sizes for the transitions as observed are 1264x620 -> 1280x659 -> 1920x1019, but we would expect
     * the direct transition 1264x620 -> 1920x1019.
     * I'm unsure what state the window is in for the intermediate transition, and why the transition can even
     * happen in two steps. Shouldn't the EDT handle all the changes in showFullscreen in one go? Why is the size
     * change delayed and "animated" over several frames/callbacks to the EDT?
     * <p>
     * FIXME: Memory resources are not very efficiently cleaned up, even after .flush() on the buffers.
     * Calling System.gc() after releasing the buffers keeps the memory usage down.
     * .
     * Should we maybe have the Main thread perform these calls, since it has nothing else to do?
     * It doesn't really matter if the GC is a stop-the-world kind, but otherwise might make sense.
     */
//    static int counter = 0;
    private void validateBuffers()
    {
//        ++counter;

        // Don't reallocate if size is 0 or less in any dimension, or we are not visible.
        if (this.getWidth() <= 0 || this.getHeight() <= 0 || !this.isVisible())
        {
            return;
        }

        // Reallocate if this is the first render (buffers are null).
        if (this.domainBackgroundBuffer == null)
        {
            this.domainBackgroundBuffer = this.getGraphicsConfiguration()
                                              .createCompatibleVolatileImage(this.getWidth(), this.getHeight());
            this.stateBackgroundBuffer = this.getGraphicsConfiguration()
                                             .createCompatibleVolatileImage(this.getWidth(), this.getHeight());
            this.stateTransitionBuffer = this.getGraphicsConfiguration()
                                             .createCompatibleVolatileImage(this.getWidth(), this.getHeight());
            this.domainBackgroundGraphics = this.domainBackgroundBuffer.createGraphics();
            this.stateBackgroundGraphics = this.stateBackgroundBuffer.createGraphics();
            this.stateTransitionGraphics = this.stateTransitionBuffer.createGraphics();
            this.domainBackgroundGraphics.setRenderingHint(KEY_TEXT_ANTIALIASING, VALUE_TEXT_ANTIALIAS_ON);
            this.domainBackgroundGraphics.setRenderingHint(KEY_ANTIALIASING, VALUE_ANTIALIAS_ON);
            this.stateBackgroundGraphics.setRenderingHint(KEY_TEXT_ANTIALIASING, VALUE_TEXT_ANTIALIAS_ON);
            this.stateBackgroundGraphics.setRenderingHint(KEY_ANTIALIASING, VALUE_ANTIALIAS_ON);
            this.stateTransitionGraphics.setRenderingHint(KEY_TEXT_ANTIALIASING, VALUE_TEXT_ANTIALIAS_ON);
            this.stateTransitionGraphics.setRenderingHint(KEY_ANTIALIASING, VALUE_ANTIALIAS_ON);
            this.requireFullRender = true;

            // FIXME:
//            System.err.println("Allocated buffers:");
//            System.err.println("Device:    " + this.getGraphicsConfiguration().getDevice().getDisplayMode().getWidth()
//                               + "x" + this.getGraphicsConfiguration().getDevice().getDisplayMode().getHeight());
//            System.err.println("Panel:     " + this.getWidth() + "x" + this.getHeight());
//            System.err.println("Buffer: " + this.domainBackgroundBuffer.getWidth() + "x" + this.domainBackgroundBuffer.getHeight());
//            System.err.println("Graphics:  " + this.getGraphicsConfiguration().getBounds());
//            System.err.println("DefaultTx: " + this.getGraphicsConfiguration().getDefaultTransform());
//            System.err.println("NormTx:    " + this.getGraphicsConfiguration().getNormalizingTransform());

//            System.out.println("" + counter + ": " + this.getWidth() + "x" + this.getHeight());
//            System.out.println("" + counter + ": First allocation of buffers.");
        }

        // Validate buffers.
        int status1 = this.domainBackgroundBuffer.validate(this.getGraphicsConfiguration());
        int status2 = this.stateBackgroundBuffer.validate(this.getGraphicsConfiguration());
        int status3 = this.stateTransitionBuffer.validate(this.getGraphicsConfiguration());
        boolean revalidate = false;

        // Reallocate if any buffers were incompatible with their graphics configuration (e.g. window moved to new
        // display device).
        // Reallocate if the panel has changed size.
        if (status1 == VolatileImage.IMAGE_INCOMPATIBLE ||
            status2 == VolatileImage.IMAGE_INCOMPATIBLE ||
            status3 == VolatileImage.IMAGE_INCOMPATIBLE ||
            this.getWidth() != this.domainBackgroundBuffer.getWidth() ||
            this.getHeight() != this.domainBackgroundBuffer.getHeight())
        {
            this.domainBackgroundGraphics.dispose();
            this.stateBackgroundGraphics.dispose();
            this.stateTransitionGraphics.dispose();
            this.domainBackgroundBuffer.flush();
            this.stateBackgroundBuffer.flush();
            this.stateTransitionBuffer.flush();
            this.domainBackgroundBuffer = this.getGraphicsConfiguration()
                                              .createCompatibleVolatileImage(this.getWidth(), this.getHeight());
            this.stateBackgroundBuffer = this.getGraphicsConfiguration()
                                             .createCompatibleVolatileImage(this.getWidth(), this.getHeight());
            this.stateTransitionBuffer = this.getGraphicsConfiguration()
                                             .createCompatibleVolatileImage(this.getWidth(), this.getHeight());
            this.domainBackgroundGraphics = this.domainBackgroundBuffer.createGraphics();
            this.stateBackgroundGraphics = this.stateBackgroundBuffer.createGraphics();
            this.stateTransitionGraphics = this.stateTransitionBuffer.createGraphics();
            this.domainBackgroundGraphics.setRenderingHint(KEY_TEXT_ANTIALIASING, VALUE_TEXT_ANTIALIAS_ON);
            this.domainBackgroundGraphics.setRenderingHint(KEY_ANTIALIASING, VALUE_ANTIALIAS_ON);
            this.stateBackgroundGraphics.setRenderingHint(KEY_TEXT_ANTIALIASING, VALUE_TEXT_ANTIALIAS_ON);
            this.stateBackgroundGraphics.setRenderingHint(KEY_ANTIALIASING, VALUE_ANTIALIAS_ON);
            this.stateTransitionGraphics.setRenderingHint(KEY_TEXT_ANTIALIASING, VALUE_TEXT_ANTIALIAS_ON);
            this.stateTransitionGraphics.setRenderingHint(KEY_ANTIALIASING, VALUE_ANTIALIAS_ON);
            this.requireFullRender = true;
            revalidate = true;

            // FIXME:
//            System.err.println("Allocated buffers:");
//            System.err.println("Device:    " + this.getGraphicsConfiguration().getDevice().getDisplayMode().getWidth()
//                               + "x" + this.getGraphicsConfiguration().getDevice().getDisplayMode().getHeight());
//            System.err.println("Panel:     " + this.getWidth() + "x" + this.getHeight());
//            System.err.println("Graphics:  " + this.getGraphicsConfiguration().getBounds());
//            System.err.println("DefaultTx: " + this.getGraphicsConfiguration().getDefaultTransform());
//            System.err.println("NormTx:    " + this.getGraphicsConfiguration().getNormalizingTransform());

            // FIXME: Required to keep memory usage down after DomainPanels reallocate their image buffers.
            System.gc();

//            System.out.println("" + counter + ": " + this.getWidth() + "x" + this.getHeight());
//            System.out.println("" + counter + ": Reallocated buffers.");
        }

        // We have to revalidate if we just reallocated the buffers, otherwise we will miss a frame.
        // (Because .validate() does not restore an image if it returns IMAGE_INCOMPATIBLE.)
        // Since we have just checked compatibility with the graphics configuration, we will skip that check here.
        if (revalidate)
        {
            status1 = this.domainBackgroundBuffer.validate(null);
            status2 = this.stateBackgroundBuffer.validate(null);
            status3 = this.stateTransitionBuffer.validate(null);

//            System.out.println("" + counter + ": Revalidated buffers.");
        }

        // If buffers were restored, require a full render.
        if (status1 == VolatileImage.IMAGE_RESTORED ||
            status2 == VolatileImage.IMAGE_RESTORED ||
            status3 == VolatileImage.IMAGE_RESTORED)
        {
            this.requireFullRender = true;

//            System.out.println("" + counter + ": Buffers restored.");
        }
    }

    /**
     * Assumes the buffers are valid and of appropriate sizes.
     * Call validateBuffers() first to validate and restore/reallocate buffers as necessary.
     */
    private void renderDomainBackground()
    {
        this.domain.renderDomainBackground(
                this.domainBackgroundGraphics,
                this.domainBackgroundBuffer.getWidth(),
                this.domainBackgroundBuffer.getHeight());
    }

    /**
     * Assumes the domainBackgroundBuffer is up-to-date. Call renderDomainBackground() first if not.
     */
    private void renderStateBackground(int stateID)
    {
        this.stateBackgroundGraphics.drawImage(this.domainBackgroundBuffer, 0, 0, null);
        this.domain.renderStateBackground(this.stateBackgroundGraphics, stateID);
    }

    /**
     * Assumes the stateBackgroundBuffer is up-to-date. Call renderStateBackground() first if not.
     */
    private void renderStateTransition(int stateID, double interpolation)
    {
        this.stateTransitionGraphics.drawImage(this.stateBackgroundBuffer, 0, 0, null);
        this.domain.renderStateTransition(this.stateTransitionGraphics, stateID, interpolation);
    }

    /**
     * The rendering thread waits here until signalRenderBegin() is called.
     */
    private synchronized void waitRenderBegin()
    {
        while (!this.signal)
        {
            try
            {
                this.wait();
            }
            catch (InterruptedException ignored)
            {
            }
        }
    }

    /**
     * The rendering thread calls this to release the EDT waiting on waitRenderFinish().
     */
    private synchronized void signalRenderFinish()
    {
        this.signal = false;
        this.notifyAll();
    }

    /**
     * The rendering loop for this DomainPanel's rendering thread.
     * The thread is started in the constructur, and runs until shutdown is signaled.
     */
    private void renderLoop()
    {
        Server.printDebug("Thread started.");

        while (true)
        {
            // The EDT's call to signalDomainRender() has a happens-before relationship with this thread's call below.
            this.waitRenderBegin();

            if (this.shutdown)
            {
                break;
            }

            // Render only as much as is necessary.
            this.isNewlyRendered = true;
            int curState = (int) this.currentStateInterpolation;
            if (this.requireFullRender)
            {
//                System.out.println("Full render.");
                this.renderDomainBackground();
                this.renderStateBackground(curState);
                this.renderStateTransition(curState, this.currentStateInterpolation - curState);
                this.requireFullRender = false;
            }
            else if ((int) this.lastStateInterpolation != curState)
            {
//                System.out.println("State render.");
                this.renderStateBackground(curState);
                this.renderStateTransition(curState, this.currentStateInterpolation - curState);
            }
            else if (this.lastStateInterpolation != this.currentStateInterpolation)
            {
//                System.out.println("Interpolation render.");
                this.renderStateTransition(curState, this.currentStateInterpolation - curState);
            }
            else
            {
                this.isNewlyRendered = false;
            }

            this.isNewlyRendered &= !this.stateTransitionBuffer.contentsLost();
            this.lastStateInterpolation = this.currentStateInterpolation;

            // This thread's call below has a happens-before relationship to the EDT's call to waitForFinishRender().
            this.signalRenderFinish();
        }

        Server.printDebug("Thread shut down.");
    }
}
