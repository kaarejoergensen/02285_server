which mvn
mvn clean install
rm -rf logs/basic/*
rm -rf logs/randomwalk/*
rm -rf logs/dungeon/*

java -jar target/02285_server-1.0-SNAPSHOT-executable.jar -c "java -Xmx32G -classpath target/classes searchclient.SearchClient -bfs" -l src/main/resources/levels/generated/basic/Easy_1 -t 1200 -o logs/basic/Easy.zip
java -jar target/02285_server-1.0-SNAPSHOT-executable.jar -c "java -Xmx32G -classpath target/classes searchclient.SearchClient -bfs" -l src/main/resources/levels/generated/basic/Easy_2 -t 1200 -o logs/basic/Medium.zip
java -jar target/02285_server-1.0-SNAPSHOT-executable.jar -c "java -Xmx32G -classpath target/classes searchclient.SearchClient -bfs" -l src/main/resources/levels/generated/basic/Easy_3 -t 1200 -o logs/basic/Hard.zip

java -jar target/02285_server-1.0-SNAPSHOT-executable.jar -c "java -Xmx32G -classpath target/classes searchclient.SearchClient -bfs" -l src/main/resources/levels/generated/randomwalk/Easy_1 -t 1200 -o logs/randomwalk/Easy.zip
java -jar target/02285_server-1.0-SNAPSHOT-executable.jar -c "java -Xmx32G -classpath target/classes searchclient.SearchClient -bfs" -l src/main/resources/levels/generated/randomwalk/Easy_2 -t 1200 -o logs/randomwalk/Medium.zip
java -jar target/02285_server-1.0-SNAPSHOT-executable.jar -c "java -Xmx32G -classpath target/classes searchclient.SearchClient -bfs" -l src/main/resources/levels/generated/randomwalk/Easy_3 -t 1200 -o logs/randomwalk/Hard.zip

java -jar target/02285_server-1.0-SNAPSHOT-executable.jar -c "java -Xmx32G -classpath target/classes searchclient.SearchClient -bfs" -l src/main/resources/levels/generated/dungeon/Easy_1 -t 1200 -o logs/dungeon/Easy.zip
java -jar target/02285_server-1.0-SNAPSHOT-executable.jar -c "java -Xmx32G -classpath target/classes searchclient.SearchClient -bfs" -l src/main/resources/levels/generated/dungeon/Easy_2 -t 1200 -o logs/dungeon/Medium.zip
java -jar target/02285_server-1.0-SNAPSHOT-executable.jar -c "java -Xmx32G -classpath target/classes searchclient.SearchClient -bfs" -l src/main/resources/levels/generated/dungeon/Easy_3 -t 1200 -o logs/dungeon/Hard.zip