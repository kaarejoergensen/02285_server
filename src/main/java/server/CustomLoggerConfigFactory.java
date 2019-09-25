package server;

import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.core.LoggerContext;
import org.apache.logging.log4j.core.appender.ConsoleAppender;
import org.apache.logging.log4j.core.config.Configuration;
import org.apache.logging.log4j.core.config.ConfigurationFactory;
import org.apache.logging.log4j.core.config.ConfigurationSource;
import org.apache.logging.log4j.core.config.Order;
import org.apache.logging.log4j.core.config.builder.api.AppenderComponentBuilder;
import org.apache.logging.log4j.core.config.builder.api.ConfigurationBuilder;
import org.apache.logging.log4j.core.config.builder.api.CustomLevelComponentBuilder;
import org.apache.logging.log4j.core.config.builder.impl.BuiltConfiguration;
import org.apache.logging.log4j.core.config.plugins.Plugin;

import java.net.URI;

@Plugin(name = "CustomLoggerConfigFactory", category = ConfigurationFactory.CATEGORY)
@Order(50)
public class CustomLoggerConfigFactory extends ConfigurationFactory {
    public static Level messageLevel = Level.forName("MESSAGE", 350);
    static Configuration createConfiguration(final String name, ConfigurationBuilder<BuiltConfiguration> builder) {
//        boolean PRINT_DEBUG = "true".equalsIgnoreCase(System.getenv("AIMAS_SERVER_DEBUG"));
        boolean PRINT_DEBUG = true;
        builder.setConfigurationName(name);
        builder.setStatusLevel(Level.WARN);
        AppenderComponentBuilder appenderBuilder = builder.newAppender("Stdout", "CONSOLE").
                addAttribute("target", ConsoleAppender.Target.SYSTEM_OUT);
        appenderBuilder.add(builder.newLayout("PatternLayout").
                addAttribute("pattern", "[%logger{36}][%level{lowerCase=true}] %msg%n").
                addAttribute("alwaysWriteExceptions", PRINT_DEBUG));
        builder.add(appenderBuilder);
        builder.add(builder.newRootLogger(PRINT_DEBUG ? Level.DEBUG : Level.INFO).add(builder.newAppenderRef("Stdout")));
        CustomLevelComponentBuilder customLevelComponentBuilder = builder.newCustomLevel(messageLevel.name(), messageLevel.intLevel());
        builder.add(customLevelComponentBuilder);

        return builder.build();
    }

    @Override
    public Configuration getConfiguration(final LoggerContext loggerContext, final ConfigurationSource source) {
        return getConfiguration(loggerContext, source.toString(), null);
    }

    @Override
    public Configuration getConfiguration(final LoggerContext loggerContext, final String name, final URI configLocation) {
        ConfigurationBuilder<BuiltConfiguration> builder = newConfigurationBuilder();
        return createConfiguration(name, builder);
    }

    @Override
    protected String[] getSupportedTypes() {
        return new String[] {"*"};
    }
}
