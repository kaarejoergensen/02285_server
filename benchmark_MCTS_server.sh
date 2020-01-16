export JAVA_HOME=~/Downloads/jdk-12.0.2
java=~/Downloads/jdk-12.0.2/bin/java
~/apache-maven-3.5.0/bin/mvn clean install
rm -rf logs/mcts/basic/*
rm -rf logs/mcts/randomwalk/*
rm -rf logs/mcts/dungeon/*

$java -jar target/02285_server-1.0-SNAPSHOT-executable.jar -c "$java -Xmx48G -classpath target/classes searchclient.SearchClient -basic" -l src/main/resources/levels/generated/basic/Easy_1 -t 1200 -o logs/mcts/basic/Easy.zip
$java -jar target/02285_server-1.0-SNAPSHOT-executable.jar -c "$java -Xmx48G -classpath target/classes searchclient.SearchClient -basic" -l src/main/resources/levels/generated/basic/Easy_2 -t 1200 -o logs/mcts/basic/Medium.zip
$java -jar target/02285_server-1.0-SNAPSHOT-executable.jar -c "$java -Xmx48G -classpath target/classes searchclient.SearchClient -basic" -l src/main/resources/levels/generated/basic/Easy_3 -t 1200 -o logs/mcts/basic/Hard.zip

$java -jar target/02285_server-1.0-SNAPSHOT-executable.jar -c "$java -Xmx48G -classpath target/classes searchclient.SearchClient -basic" -l src/main/resources/levels/generated/randomwalk/Easy_1 -t 1200 -o logs/mcts/randomwalk/Easy.zip
$java -jar target/02285_server-1.0-SNAPSHOT-executable.jar -c "$java -Xmx48G -classpath target/classes searchclient.SearchClient -basic" -l src/main/resources/levels/generated/randomwalk/Easy_2 -t 1200 -o logs/mcts/randomwalk/Medium.zip
$java -jar target/02285_server-1.0-SNAPSHOT-executable.jar -c "$java -Xmx48G -classpath target/classes searchclient.SearchClient -basic" -l src/main/resources/levels/generated/randomwalk/Easy_3 -t 1200 -o logs/mcts/randomwalk/Hard.zip

$java -jar target/02285_server-1.0-SNAPSHOT-executable.jar -c "$java -Xmx48G -classpath target/classes searchclient.SearchClient -basic" -l src/main/resources/levels/generated/dungeon/Easy_1 -t 1200 -o logs/mcts/dungeon/Easy.zip
$java -jar target/02285_server-1.0-SNAPSHOT-executable.jar -c "$java -Xmx48G -classpath target/classes searchclient.SearchClient -basic" -l src/main/resources/levels/generated/dungeon/Easy_2 -t 1200 -o logs/mcts/dungeon/Medium.zip
$java -jar target/02285_server-1.0-SNAPSHOT-executable.jar -c "$java -Xmx48G -classpath target/classes searchclient.SearchClient -basic" -l src/main/resources/levels/generated/dungeon/Easy_3 -t 1200 -o logs/mcts/dungeon/Hard.zip
