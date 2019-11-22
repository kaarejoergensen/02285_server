package nn.impl;

import nn.NNet;
import searchclient.State;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.List;

public class PythonNNet extends NNet {
    private static final String PYTHON_PATH = "python";
    private static final String SCRIPT_PATH = "./src/main/python/main.py";

    private BufferedReader clientReader;
    private BufferedWriter clientWriter;

    public PythonNNet() throws IOException {
        this.run();
    }

    private void run() throws IOException {
        ProcessBuilder processBuilder = new ProcessBuilder(PYTHON_PATH, SCRIPT_PATH).redirectError(ProcessBuilder.Redirect.INHERIT);
        Process process = processBuilder.start();

        InputStream clientIn = process.getInputStream();
        BufferedInputStream outputFromClient = clientIn instanceof BufferedInputStream ?
                (BufferedInputStream) clientIn :
                new BufferedInputStream(clientIn);
        OutputStream clientOut = process.getOutputStream();
        BufferedOutputStream inputToClient = clientOut instanceof BufferedOutputStream ?
                (BufferedOutputStream) clientOut :
                new BufferedOutputStream(clientOut);

        this.clientReader = new BufferedReader(new InputStreamReader(outputFromClient, StandardCharsets.UTF_8));
        this.clientWriter = new BufferedWriter(new OutputStreamWriter(inputToClient, StandardCharsets.UTF_8));
    }

    @Override
    public void train(List<String[]> examples) {
        this.writeToPython("train", examples.toString());
        this.readFromPython();
    }

    @Override
    public float predict(State state) {
        this.writeToPython("predict", state.toMLString());
        return Float.parseFloat(this.readFromPython());
    }

    @Override
    public void saveModel(Path fileName) {
        this.writeToPython("saveModel", fileName.toString());
        this.readFromPython();
    }

    @Override
    public void loadModel(Path fileName) {
        this.writeToPython("loadModel", fileName.toString());
        this.readFromPython();
    }

    @Override
    public void close() throws IOException {
        this.writeToPython("close", "");
        this.readFromPython();
        this.clientWriter.close();
        this.clientReader.close();
    }

    private String readFromPython() {
        try {
            String line = this.clientReader.readLine();
            if (line == null) {
                System.err.println("Read from python failed: null line received");
                System.exit(-1);
            }
            return line;
        } catch (IOException e) {
            System.err.println("Read from python failed: " + e.getMessage());
            System.exit(-1);
        }
        return "";
    }

    private void writeToPython(String method, String args) {
        try {
            this.clientWriter.write(method + System.lineSeparator());
            this.clientWriter.flush();
            this.clientWriter.write(args + System.lineSeparator() + System.lineSeparator());
            this.clientWriter.flush();
        } catch (IOException e) {
            System.err.println("Write to python failed: " + e.getMessage());
            System.exit(-1);
        }
    }
}
