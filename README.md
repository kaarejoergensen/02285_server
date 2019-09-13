## 02285_Server

1. `mvn clean install`

2.1. Windows:
`java -jar .\target\02285_server-1.0-SNAPSHOT-executable.jar -c "java -classpath .\target\classes searchclient.SearchClient -dfs" -l .\src\main\resources\levels\single_agent\SAFriendOfDFS.lvl -t 120 -g`

2.2. Mac:
`java -jar ./target/02285_server-1.0-SNAPSHOT-executable.jar -c "java -classpath ./target/classes searchclient.SearchClient -dfs" -l ./src/main/resources/levels/single_agent/SAFriendOfDFS.lvl -t 120 -g`
