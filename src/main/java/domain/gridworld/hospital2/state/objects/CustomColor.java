package domain.gridworld.hospital2.state.objects;

import lombok.Data;
import lombok.RequiredArgsConstructor;

import java.awt.*;

@Data
@RequiredArgsConstructor
public class CustomColor {
    private final Color current;
    private Color next;
}
