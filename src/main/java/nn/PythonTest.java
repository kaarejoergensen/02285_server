package nn;


public class PythonTest {



    public static void main(String[] args) {
        PythonRunner r = new PythonRunner("neural.py");
        System.out.println("Trying to run: " + r.getFullPath());
        String[] a = new String[1];
        a[0] = "Kåre";
        r.run(a);

    }


}
