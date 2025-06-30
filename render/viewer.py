import pygame


class Viewer:
    def __init__(self, options):
        """
        Initialize the viewer with the given options

        Params:
            (dict) options: Values for the viewer
        """
        ## Initialize pygame
        pygame.init()

        ## Unpack the options
        self.field_width = options["field_width"]
        self.field_height = options["field_height"]
        self.agent_size = options["agent_size"]
        self.ball_size = options["ball_size"]
        self.goal_zone = options["goal_zone"]
        self.goal_size = options["goal_size"]
        self.goal_box_thickness = 10

        ## Define the rectangles that make up the goalpost
        self.goal_top_rect = pygame.Rect(
            self.goal_zone["top_left"][0] - self.goal_box_thickness,
            self.goal_zone["top_left"][1] - self.goal_box_thickness,
            self.goal_size[0] + self.goal_box_thickness,
            self.goal_box_thickness,
        )

        self.goal_bottom_rect = pygame.Rect(
            self.goal_zone["bottom_left"][0] - self.goal_box_thickness,
            self.goal_zone["bottom_left"][1],
            self.goal_size[0] + self.goal_box_thickness,
            self.goal_box_thickness,
        )

        self.goal_middle_rect = pygame.Rect(
            self.goal_zone["top_left"][0] - self.goal_box_thickness,
            self.goal_zone["top_left"][1],
            self.goal_box_thickness,
            self.goal_size[1],
        )

        self.goal_inner_rect = pygame.Rect(
            self.goal_zone["top_left"][0],
            self.goal_zone["top_left"][1],
            self.goal_size[0],
            self.goal_size[1],
        )

        ## Initialize the screen and clock
        self.screen = pygame.display.set_mode((self.field_width, self.field_height))
        self.clock = pygame.time.Clock()

        ## Load the image resources
        self._load_resources()

    def close(self):
        """
        Close the viewer
        """
        ## Destroy pygame
        pygame.quit()

    def render(self, agent_positions, ball_position):
        """
        Render the env with the given positions of agents and ball position

        Params:
            (dict)     agent_positions: Key value pair of agent name and their position
            (float[2]) ball_position  : Position vector of the ball
        """
        ## Draw the ground image
        self.screen.blit(self.football_pitch_image, (0, 0))

        ## Draw goal
        pygame.draw.rect(self.screen, "white", self.goal_bottom_rect)
        pygame.draw.rect(self.screen, "white", self.goal_top_rect)
        pygame.draw.rect(self.screen, "white", self.goal_middle_rect)
        pygame.draw.rect(self.screen, "#aaaaaa", self.goal_inner_rect)

        ## Draw agents
        for _, position in agent_positions.items():
            self.screen.blit(
                self.agent_image, position - (self.agent_image.get_width() / 2)
            )

        ## Draw football
        self.screen.blit(
            self.football_image, ball_position - (self.football_image.get_width() / 2)
        )

        ## Update display
        pygame.display.flip()

    def _load_resources(self):
        """
        Load the image resources
        """
        ## Load the pitch image and scale it to required size
        football_pitch_image = pygame.image.load("resources/football_pitch.png")
        self.football_pitch_image = pygame.transform.scale(
            football_pitch_image, (self.field_width, self.field_height)
        )

        ## Load the football image and scale it to required size
        football_image = pygame.image.load("resources/football.png")
        self.football_image = pygame.transform.scale(
            football_image, (2 * self.ball_size, 2 * self.ball_size)
        )

        ## Load the agent image and scale it to required size
        agent_image = pygame.image.load("resources/player.png")
        self.agent_image = pygame.transform.scale(
            agent_image, (2 * self.agent_size, 2 * self.agent_size)
        )
