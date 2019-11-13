package nn;


public class PythonTest {



    public static void main(String[] args) {
        PythonRunner r = new PythonRunner("test.py");
        String[] a = new String[1];
        a[0] = "Kåre";
        r.run(a);

        a[0] = "Fredrik";
        r.run(a);
    }


}
