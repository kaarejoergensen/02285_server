package levelgenerator.modules;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;

/*
    https://www.journaldev.com/937/compile-run-java-program-another-java-program
    Credits: Pankai
 */

public class LevelTester {


    private String argument;
    private String endArgument;

    private String timeout;

    public LevelTester(String algorithm, String timeout, String path){
        argument = "java -jar ./target/02285_server-1.0-SNAPSHOT-executable.jar -c \"java -Xmx2G -classpath target/classes ./target/classes searchclient.SearchClient -";
        argument += algorithm + "\" -l ";
        argument += path;
        endArgument = ".lvl -t " + timeout + " -g";

        System.out.println("Fult Argument: " + argument + "{LEVEL}" + endArgument);
    }

    public void runLevel(String level) throws Exception{
        Process pro = Runtime.getRuntime().exec(argument + level + endArgument);
        printOutput(level + " stdout: ", pro.getInputStream());
        printOutput(level + " stderr:" , pro.getErrorStream());
        pro.waitFor();
        System.out.println(level + " exitValue() " + pro.exitValue());
    }



    public void printOutput(String cmd, InputStream input) throws  Exception{
        String line = null;
        BufferedReader in = new BufferedReader(
                new InputStreamReader(input));
        while ((line = in.readLine()) != null) {
            System.out.println(cmd + " " + line);
        }
    }

}
