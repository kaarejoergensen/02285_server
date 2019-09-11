package domain;

public class ParseException
        extends Exception
{
    public final int lineNumber;

    public ParseException(String message)
    {
        super(message);
        this.lineNumber = -1;
    }

    public ParseException(String message, int lineNumber)
    {
        super(message);
        this.lineNumber = lineNumber;
    }

    @Override
    public String getMessage()
    {
        if (this.lineNumber == -1)
        {
            return super.getMessage();
        }
        else
        {
            return "On line " + this.lineNumber + ": " + super.getMessage();
        }
    }
}
