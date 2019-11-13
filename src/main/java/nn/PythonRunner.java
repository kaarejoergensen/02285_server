package nn;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class PythonRunner {

    private final String PATH = "./src/main/java/nn/";

    private String filename;

    private ProcessBuilder process;

    private BufferedReader output;
    private BufferedReader errors;

    public PythonRunner(String filename){
        this.filename = filename;
    }

    public void run(String[] args){
        process = new ProcessBuilder("python", getFullPath()); //Save the instance
        //Attach arguments
        for(String s : args){
            process.command().add(s);
        }

        try {
            Process p = process.start();

            output = new BufferedReader(new InputStreamReader(p.getInputStream()));
            errors = new BufferedReader(new InputStreamReader(p.getErrorStream()));

            int exitValue = p.waitFor();

            if(exitValue != 0){
                System.out.println("Error in script '" + filename + "'.");
                printStream(errors);
            }

            printStream(output);

        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
    }


    private void printStream(BufferedReader r) throws IOException {
        String strCurrentLine = null;
        while ((strCurrentLine = r.readLine()) != null) {
            System.out.println(strCurrentLine);
        }
    }

    public String getFullPath(){
        return PATH + filename;
    }


}
