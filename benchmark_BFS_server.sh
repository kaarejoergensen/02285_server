export JAVA_HOME=~/Downloads/jdk-12.0.2
java=~/Downloads/jdk-12.0.2/bin/java
~/apache-maven-3.5.0/bin/mvn clean install
mkdir -p logs/bfs/basic
mkdir -p logs/bfs/randomwalk
mkdir -p logs/bfs/dungeon

rm -rf logs/bfs/basic/*
rm -rf logs/bfs/randomwalk/*
rm -rf logs/bfs/dungeon/*

$java -jar target/02285_server-1.0-SNAPSHOT-executable.jar -c "$java -Xmx70G -classpath target/classes searchclient.SearchClient -bfs" -l src/main/resources/levels/generated/basic/Easy_1 -t 1200 -o logs/bfs/basic/Easy.zip
$java -jar target/02285_server-1.0-SNAPSHOT-executable.jar -c "$java -Xmx70G -classpath target/classes searchclient.SearchClient -bfs" -l src/main/resources/levels/generated/basic/Easy_2 -t 1200 -o logs/bfs/basic/Medium.zip
$java -jar target/02285_server-1.0-SNAPSHOT-executable.jar -c "$java -Xmx70G -classpath target/classes searchclient.SearchClient -bfs" -l src/main/resources/levels/generated/basic/Easy_3 -t 1200 -o logs/bfs/basic/Hard.zip

$java -jar target/02285_server-1.0-SNAPSHOT-executable.jar -c "$java -Xmx70G -classpath target/classes searchclient.SearchClient -bfs" -l src/main/resources/levels/generated/randomwalk/Easy_1 -t 1200 -o logs/bfs/randomwalk/Easy.zip
$java -jar target/02285_server-1.0-SNAPSHOT-executable.jar -c "$java -Xmx70G -classpath target/classes searchclient.SearchClient -bfs" -l src/main/resources/levels/generated/randomwalk/Easy_2 -t 1200 -o logs/bfs/randomwalk/Medium.zip
$java -jar target/02285_server-1.0-SNAPSHOT-executable.jar -c "$java -Xmx70G -classpath target/classes searchclient.SearchClient -bfs" -l src/main/resources/levels/generated/randomwalk/Easy_3 -t 1200 -o logs/bfs/randomwalk/Hard.zip

$java -jar target/02285_server-1.0-SNAPSHOT-executable.jar -c "$java -Xmx70G -classpath target/classes searchclient.SearchClient -bfs" -l src/main/resources/levels/generated/dungeon/Easy_1 -t 1200 -o logs/bfs/dungeon/Easy.zip
$java -jar target/02285_server-1.0-SNAPSHOT-executable.jar -c "$java -Xmx70G -classpath target/classes searchclient.SearchClient -bfs" -l src/main/resources/levels/generated/dungeon/Easy_2 -t 1200 -o logs/bfs/dungeon/Medium.zip
$java -jar target/02285_server-1.0-SNAPSHOT-executable.jar -c "$java -Xmx70G -classpath target/classes searchclient.SearchClient -bfs" -l src/main/resources/levels/generated/dungeon/Easy_3 -t 1200 -o logs/bfs/dungeon/Hard.zip
