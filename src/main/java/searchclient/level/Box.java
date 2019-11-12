package searchclient.level;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import shared.Farge;

@RequiredArgsConstructor
@Getter
public class Box {
    private final Character character;
    private final Farge color;
}
