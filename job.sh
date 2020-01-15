export JAVA_HOME=~/Downloads/jdk-12.0.2
cat src/main/resources/levels/ml/01.lvl | ~/apache-maven-3.5.0/bin/mvn clean compile exec:java -Dexec.mainClass="searchclient.SearchClient" -Dexec.args="-alpha -train -python python3"
