# Learning and Planning with Action Model Learning

## 02285_Server run details

1. `mvn clean install`

2.1. Windows:
`java -jar .\target\02285_server-1.0-SNAPSHOT-executable.jar -c "java -classpath .\target\classes searchclient.SearchClient -dfs" -l .\src\main\resources\levels\single_agent\SAFriendOfDFS.lvl -t 120 -g`

2.2. Mac:
`java -jar ./target/02285_server-1.0-SNAPSHOT.jar -c "java -classpath ./target/classes searchclient.SearchClient -dfs" -l ./src/main/resources/levels/single_agent/SAFriendOfDFS.lvl -t 120 -g`



## Remastered Features

### Wizzy Boii

* Can change colors of boxes
* Is Grey

![alt text](https://art.pixilart.com/1d891dc0dc174c0.png)


### Domain Improvements 

* GameObjects: Parent class to all objects in scene. Boxes and Agents are separated into two child classes
* Canvas is a new class describing variables and methods regards to the relativity of the program window.

## Todo
* Clean shutdown of server on Ctrl-C, SIGINT, or whatever mechanisms that cause the JVM to shut down nicely.

## Nice-To-Have
* DOMAIN: More implementations.
* GUI: More speed presets?
* GUI: Help overlay.
* GUI: Reloading/navigating new client/domains without server restart?