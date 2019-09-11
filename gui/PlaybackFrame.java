package gui;

import domain.Domain;
import gui.widgets.DomainPanel;
import gui.widgets.PlayPauseButton;
import gui.widgets.SeekBar;
import gui.widgets.SkipBackwardButton;
import gui.widgets.SkipForwardButton;
import gui.widgets.StepBackwardButton;
import gui.widgets.StepForwardButton;

import javax.swing.AbstractAction;
import javax.swing.BorderFactory;
import javax.swing.JComponent;
import javax.swing.JFormattedTextField;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.KeyStroke;
import javax.swing.WindowConstants;
import javax.swing.text.NumberFormatter;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Frame;
import java.awt.GraphicsConfiguration;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.GridLayout;
import java.awt.Insets;
import java.awt.Point;
import java.awt.Toolkit;
import java.awt.event.ActionEvent;
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;
import java.awt.event.FocusAdapter;
import java.awt.event.FocusEvent;
import java.awt.event.InputEvent;
import java.awt.event.KeyEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.text.NumberFormat;

class PlaybackFrame
        extends JFrame
{
    private Point previousWindowLocation = null;
    private Dimension previousWindowSize = null;
    private int previousWindowExtendedState = -1;

    private JLabel levelNameLabel;
    private JLabel clientNameLabel;
    private JLabel shownStateLabel;
    private JLabel shownStateTimeLabel;

    private DomainPanel domainPanel;
    private JPanel botPanel;

    private JFormattedTextField speedField;
    private SeekBar seekBar;
    private PlayPauseButton playPauseButton;

    PlaybackFrame(PlaybackManager playbackManager, Domain domain, GraphicsConfiguration gc)
    {
        super("AI&MAS Planning Domain Server", gc);

        /*
            A LITTLE CUSTOMIZATION
        */
        Color backgroundColor = Color.LIGHT_GRAY;
        Color borderColor = Color.DARK_GRAY;
        Color foregroundColor = Color.BLACK;
        int buttonSize = 35;
        int buttonMargin = 5;

        /*
            TOP PANEL
         */
        JPanel topPanel = new JPanel();
        topPanel.setLayout(new GridLayout(1, 4));
        topPanel.setBackground(backgroundColor);
        topPanel.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createMatteBorder(0, 0, 1, 0, borderColor),
                BorderFactory.createEmptyBorder(2, 5, 2, 5)));
        this.add(topPanel, BorderLayout.PAGE_START);


        this.levelNameLabel = new JLabel("Level: " + domain.getLevelName());
        final Font labelFont = this.levelNameLabel.getFont().deriveFont(Font.BOLD, 16);
        this.levelNameLabel.setFont(labelFont);
        topPanel.add(this.levelNameLabel);

        this.clientNameLabel = new JLabel("Client: ");
        this.clientNameLabel.setFont(labelFont);
        topPanel.add(this.clientNameLabel);

        this.shownStateLabel = new JLabel("State: ");
        this.shownStateLabel.setFont(labelFont);
        topPanel.add(this.shownStateLabel);

        this.shownStateTimeLabel = new JLabel("State time: ");
        this.shownStateTimeLabel.setFont(labelFont);
        topPanel.add(this.shownStateTimeLabel);

        /*
            LEVEL PANEL
         */
        this.domainPanel = new DomainPanel(domain);
        this.domainPanel.setFocusable(true);
        this.add(this.domainPanel, BorderLayout.CENTER);

        /*
            BOTTOM PANEL
        */
        this.botPanel = new JPanel();
        this.botPanel.setLayout(new GridBagLayout());
        this.botPanel.setBackground(backgroundColor);
        this.botPanel.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createMatteBorder(1, 0, 0, 0, borderColor),
                BorderFactory.createEmptyBorder(0, 5, 0, 5)));
        this.add(this.botPanel, BorderLayout.PAGE_END);

        JLabel speedLabel = new JLabel("Speed:");
        this.botPanel.add(speedLabel);

        var intFormat = NumberFormat.getIntegerInstance();
        intFormat.setGroupingUsed(false);
        NumberFormatter intFormatter = new NumberFormatter(intFormat);
        intFormatter.setMinimum(0);
        intFormatter.setAllowsInvalid(true);
        this.speedField = new JFormattedTextField(intFormatter)
        {
            @Override
            protected void invalidEdit()
            {
                // No beeping....
            }
        };
        this.speedField.setFocusLostBehavior(JFormattedTextField.COMMIT_OR_REVERT);
        this.speedField.setColumns(5);
        this.speedField.setValue(0);
        this.speedField.setToolTipText("ms/action");
        this.speedField.setHorizontalAlignment(JFormattedTextField.RIGHT);
        this.speedField.setMinimumSize(this.speedField.getPreferredSize());
        this.speedField.addFocusListener(new FocusAdapter()
        {
            @Override
            public void focusGained(FocusEvent e)
            {
                // This is silly, but required because the selection is otherwise immediately reset.
                PlaybackFrame.this.speedField.setText(PlaybackFrame.this.speedField.getText());
                PlaybackFrame.this.speedField.selectAll();
            }
        });
        this.speedField.addPropertyChangeListener("value", e -> playbackManager.setSpeed((int) e.getNewValue()));
        var c = new GridBagConstraints();
        c.insets = new Insets(2, 5, 2, 5);
        this.botPanel.add(this.speedField, c);

        this.seekBar = new SeekBar();
        this.seekBar.setBackground(backgroundColor);
        this.seekBar.setFocusable(false);
        c = new GridBagConstraints();
        c.fill = GridBagConstraints.BOTH;
        c.weightx = 1.0;
        this.botPanel.add(this.seekBar, c);

        SkipBackwardButton skipBackwardButton = new SkipBackwardButton(playbackManager::skipBackward);
        skipBackwardButton.setFocusable(false);
        this.botPanel.add(skipBackwardButton);
        skipBackwardButton.render(buttonSize, buttonSize, buttonMargin, backgroundColor, foregroundColor);

        StepBackwardButton stepBackwardButton = new StepBackwardButton(() -> playbackManager.stepBackward(1));
        stepBackwardButton.setFocusable(false);
        this.botPanel.add(stepBackwardButton);
        stepBackwardButton.render(buttonSize, buttonSize, buttonMargin, backgroundColor, foregroundColor);

        this.playPauseButton = new PlayPauseButton(playbackManager::togglePlayPause);
        this.playPauseButton.setFocusable(false);
        this.botPanel.add(this.playPauseButton);
        this.playPauseButton.render(buttonSize, buttonSize, buttonMargin, backgroundColor, foregroundColor);

        StepForwardButton stepForwardButton = new StepForwardButton(() -> playbackManager.stepForward(1));
        stepForwardButton.setFocusable(false);
        this.botPanel.add(stepForwardButton);
        stepForwardButton.render(buttonSize, buttonSize, buttonMargin, backgroundColor, foregroundColor);

        SkipForwardButton skipForwardButton = new SkipForwardButton(playbackManager::skipForward);
        skipForwardButton.setFocusable(false);
        this.botPanel.add(skipForwardButton);
        skipForwardButton.render(buttonSize, buttonSize, buttonMargin, backgroundColor, foregroundColor);

        /*
            SET UP HOTKEYS
         */
        int acceleratorKey = Toolkit.getDefaultToolkit().getMenuShortcutKeyMaskEx();

        // Bind inputs.
        var globalInputMap = this.getRootPane().getInputMap(JComponent.WHEN_IN_FOCUSED_WINDOW);
        globalInputMap.put(KeyStroke.getKeyStroke(KeyEvent.VK_W, acceleratorKey), "CloseWindows");
        globalInputMap.put(KeyStroke.getKeyStroke(KeyEvent.VK_F, 0), "ToggleFullscreen");
        globalInputMap.put(KeyStroke.getKeyStroke(KeyEvent.VK_I, 0), "ToggleInterface");
        globalInputMap.put(KeyStroke.getKeyStroke(KeyEvent.VK_SPACE, 0), "TogglePlayPause");
        globalInputMap.put(KeyStroke.getKeyStroke(KeyEvent.VK_LEFT, 0), "StepBackward1");
        globalInputMap.put(KeyStroke.getKeyStroke(KeyEvent.VK_RIGHT, 0), "StepForward1");
        globalInputMap.put(KeyStroke.getKeyStroke(KeyEvent.VK_LEFT, InputEvent.SHIFT_DOWN_MASK), "StepBackward10");
        globalInputMap.put(KeyStroke.getKeyStroke(KeyEvent.VK_RIGHT, InputEvent.SHIFT_DOWN_MASK), "StepForward10");
        globalInputMap.put(KeyStroke.getKeyStroke(KeyEvent.VK_LEFT, acceleratorKey), "SkipBackward");
        globalInputMap.put(KeyStroke.getKeyStroke(KeyEvent.VK_RIGHT, acceleratorKey), "SkipForward");
        globalInputMap.put(KeyStroke.getKeyStroke(KeyEvent.VK_1, 0), "SetSpeed1");
        globalInputMap.put(KeyStroke.getKeyStroke(KeyEvent.VK_2, 0), "SetSpeed2");
        globalInputMap.put(KeyStroke.getKeyStroke(KeyEvent.VK_3, 0), "SetSpeed3");
        globalInputMap.put(KeyStroke.getKeyStroke(KeyEvent.VK_4, 0), "SetSpeed4");
        globalInputMap.put(KeyStroke.getKeyStroke(KeyEvent.VK_5, 0), "SetSpeed5");

        // Bind actions.
        var globalActionMap = this.getRootPane().getActionMap();
        globalActionMap.put("CloseWindows", new AbstractAction()
        {
            @Override
            public void actionPerformed(ActionEvent e)
            {
                playbackManager.closePlaybackFrames();
            }
        });
        globalActionMap.put("ToggleFullscreen", new AbstractAction()
        {
            @Override
            public void actionPerformed(ActionEvent e)
            {
                playbackManager.toggleFullscreen();
            }
        });
        globalActionMap.put("ToggleInterface", new AbstractAction()
        {
            @Override
            public void actionPerformed(ActionEvent e)
            {
                playbackManager.toggleInterface();
            }
        });
        globalActionMap.put("TogglePlayPause", new AbstractAction()
        {
            @Override
            public void actionPerformed(ActionEvent e)
            {
                playbackManager.togglePlayPause();
            }
        });
        globalActionMap.put("StepBackward1", new AbstractAction()
        {
            @Override
            public void actionPerformed(ActionEvent e)
            {
                playbackManager.stepBackward(1);
            }
        });
        globalActionMap.put("StepForward1", new AbstractAction()
        {
            @Override
            public void actionPerformed(ActionEvent e)
            {
                playbackManager.stepForward(1);
            }
        });
        globalActionMap.put("StepBackward10", new AbstractAction()
        {
            @Override
            public void actionPerformed(ActionEvent e)
            {
                playbackManager.stepBackward(10);
            }
        });
        globalActionMap.put("StepForward10", new AbstractAction()
        {
            @Override
            public void actionPerformed(ActionEvent e)
            {
                playbackManager.stepForward(10);
            }
        });
        globalActionMap.put("SkipBackward", new AbstractAction()
        {
            @Override
            public void actionPerformed(ActionEvent e)
            {
                playbackManager.skipBackward();
            }
        });
        globalActionMap.put("SkipForward", new AbstractAction()
        {
            @Override
            public void actionPerformed(ActionEvent e)
            {
                playbackManager.skipForward();
            }
        });
        globalActionMap.put("SetSpeed1", new AbstractAction()
        {
            @Override
            public void actionPerformed(ActionEvent e)
            {
                playbackManager.setSpeed(50);
            }
        });
        globalActionMap.put("SetSpeed2", new AbstractAction()
        {
            @Override
            public void actionPerformed(ActionEvent e)
            {
                playbackManager.setSpeed(100);
            }
        });
        globalActionMap.put("SetSpeed3", new AbstractAction()
        {
            @Override
            public void actionPerformed(ActionEvent e)
            {
                playbackManager.setSpeed(250);
            }
        });
        globalActionMap.put("SetSpeed4", new AbstractAction()
        {
            @Override
            public void actionPerformed(ActionEvent e)
            {
                playbackManager.setSpeed(500);
            }
        });
        globalActionMap.put("SetSpeed5", new AbstractAction()
        {
            @Override
            public void actionPerformed(ActionEvent e)
            {
                playbackManager.setSpeed(1000);
            }
        });

        // When speed text field has focus, ignore the global 1-5 key press events.
        var speedFieldInputMap = speedField.getInputMap(JComponent.WHEN_FOCUSED);
        speedFieldInputMap.put(KeyStroke.getKeyStroke(KeyEvent.VK_1, 0), "Ignore");
        speedFieldInputMap.put(KeyStroke.getKeyStroke(KeyEvent.VK_2, 0), "Ignore");
        speedFieldInputMap.put(KeyStroke.getKeyStroke(KeyEvent.VK_3, 0), "Ignore");
        speedFieldInputMap.put(KeyStroke.getKeyStroke(KeyEvent.VK_4, 0), "Ignore");
        speedFieldInputMap.put(KeyStroke.getKeyStroke(KeyEvent.VK_5, 0), "Ignore");
        speedField.getActionMap().put("Ignore", new AbstractAction()
        {
            @Override
            public void actionPerformed(ActionEvent e)
            {
            }
        });

        /*
            FOR CLOSING WINDOWS
         */
        this.setDefaultCloseOperation(WindowConstants.DO_NOTHING_ON_CLOSE);
        this.addWindowListener(new WindowAdapter()
        {
            @Override
            public void windowClosing(WindowEvent e)
            {
                playbackManager.closePlaybackFrames();
            }
        });

        /*
            STORE WINDOW STATE
         */
        this.addComponentListener(new ComponentAdapter()
        {
            @Override
            public void componentResized(ComponentEvent e)
            {
                PlaybackFrame.this.storeFrameSizeLocation();
            }

            @Override
            public void componentMoved(ComponentEvent e)
            {
                PlaybackFrame.this.storeFrameSizeLocation();
            }
        });

        /*
            FOCUS OFF SPEED FIELD BY CLICKING ANYWHERE ELSE
         */
        var giveFocusToDomainPanel = new MouseAdapter()
        {
            @Override
            public void mousePressed(MouseEvent e)
            {
                PlaybackFrame.this.domainPanel.requestFocusInWindow();
            }
        };
        topPanel.addMouseListener(giveFocusToDomainPanel);
        this.domainPanel.addMouseListener(giveFocusToDomainPanel);
        for (var component : this.botPanel.getComponents())
        {
            if (component != this.speedField)
            {
                component.addMouseListener(giveFocusToDomainPanel);
            }
        }
    }

    /**
     * Shows the frame in borderless fullscreen mode.
     */
    void showFullscreen()
    {
        if (this.isDisplayable())
        {
            // Store size, location, and state before disposing, so we can restore when we return to windowed mode.
            this.storeFrameSizeLocation();
            this.previousWindowExtendedState = this.getExtendedState();
            this.dispose();
        }
        this.setUndecorated(true);
        this.setResizable(false);
        this.setExtendedState(Frame.MAXIMIZED_BOTH);
        this.setVisible(true);
    }

    /**
     * Shows the frame in windowed mode.
     */
    void showWindowed()
    {
        if (this.isDisplayable())
        {
            this.dispose();
        }
        this.setUndecorated(false);
        this.setResizable(true);
        if (this.previousWindowExtendedState != -1)
        {
            // Frame has been shown in windowed mode before, restore size, location, and state.
            this.setSize(this.previousWindowSize);
            this.setLocation(this.previousWindowLocation);
            this.setExtendedState(this.previousWindowExtendedState);
        }
        else
        {
            // First time frame is shown in windowed mode, set default size, location, and state.
            // FIXME: Account for hiDPI displays; use some fixed fraction of display size (e.g. 66% x 66%).
            this.setSize(1280, 720);
            this.setLocationRelativeTo(this);
            this.setExtendedState(Frame.NORMAL);
        }
        this.setVisible(true);
    }

    /**
     * Called whenever frame is shown, moved, or resized.
     * Called before going borderless fullscreen mode from windowed mode.
     * This maintains the last size and location of the frame when it was in the normal state.
     */
    private void storeFrameSizeLocation()
    {
        if (this.getExtendedState() == Frame.NORMAL)
        {
            this.previousWindowSize = this.getSize();
            this.previousWindowLocation = this.getLocation();
        }
    }

    void showInterface()
    {
        if (!this.botPanel.isVisible())
        {
            this.botPanel.setVisible(true);
        }
        this.repaint();
    }

    void hideInterface()
    {
        if (this.botPanel.isVisible())
        {
            this.botPanel.setVisible(false);
        }
        this.repaint();
    }

    void takeFocus()
    {
        this.domainPanel.requestFocus();
    }

    /**
     * Repaints the buffers to the screen if the rendering thread rendered new content in its last rendering pass.
     * IMPORTANT: Must only be called by the EDT, after waiting on waitDomainRender().
     */
    public void repaintDomainIfNewlyRendered()
    {
        this.domainPanel.repaintIfNewlyRendered();
    }

    public void startRenderingThread()
    {
        this.domainPanel.startRenderingThread();
    }

    /**
     * Signals the rendering thread for the DomainPanel to shut down.
     */
    public void shutdownRenderingThread()
    {
        this.domainPanel.shutdownRenderingThread();
    }

    /**
     * Signals the rendering thread for the DomainPanel to render the given state interpolation.
     */
    public void signalDomainRender(double stateInterpolation)
    {
        this.domainPanel.signalRenderBegin(stateInterpolation);
    }

    /**
     * Wait for the DomainPanel's rendering thread to finish rendering after the last call to signalDomainRender().
     */
    public void waitDomainRender()
    {
        this.domainPanel.waitRenderFinish();
    }

    JFormattedTextField getSpeedField()
    {
        return this.speedField;
    }

    SeekBar getSeekBar()
    {
        return this.seekBar;
    }

    void updatePlayPauseButtonIcon(boolean isPlaying)
    {
        this.playPauseButton.setCurrentIcon(isPlaying);
    }

    void setLevelName(String levelName)
    {
        this.levelNameLabel.setText("Level: " + levelName);
    }

    void setClientName(String clientName)
    {
        this.clientNameLabel.setText("Client: " + clientName);
    }

    void setShownState(String shownState)
    {
        this.shownStateLabel.setText("State: " + shownState);
    }

    void setShownStateTime(String time)
    {
        this.shownStateTimeLabel.setText("State time: " + time);
    }
}
