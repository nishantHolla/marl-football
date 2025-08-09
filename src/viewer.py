import pygame
from pathlib import Path


class Viewer:
    def __init__(self, options):
        """
        Initialize the viewer with the given options

        Params:
            (dict) options: Values for the viewer
        """
        ## Initialize pygame
        pygame.init()
        pygame.font.init()

        self.font = pygame.font.SysFont("Arial", 36)

        ## Unpack the options
        self.field_width = options["field_width"]
        self.field_height = options["field_height"]
        self.agent_size = options["agent_size"]
        self.ball_size = options["ball_size"]
        self.goal_zone = options["goal_zone"]
        self.goal_size = options["goal_size"]
        self.obstacle_size = options["obstacle_size"]
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

    def render(
        self, agent_positions, ball_position, obstacle_position, episode_number=None
    ):
        """
        Render the env with the given positions of agents and ball position

        Params:
            (dict)     agent_positions  : Key value pair of agent name and their position
            (float[2]) ball_position    : Position vector of the ball
            (float[2]) obstacle_position: Position of the obstacle
            (int)      episode_number   : Episode number to display on top left of the screen
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

        ## Draw obstacle
        self.screen.blit(
            self.obstacle_image,
            obstacle_position - (self.obstacle_image.get_width() / 2),
        )

        ## Draw episode number
        if episode_number is not None:
            text_surface = self.font.render(
                f"Episode: {episode_number}", True, (0, 0, 0)
            )
            self.screen.blit(text_surface, (10, 10))

        ## Update display
        pygame.display.flip()

    def _load_resources(self):
        """
        Load the image resources
        """

        RESOURCE_DIR = Path("../resources")

        def get_resource(resource_name):
            resource = RESOURCE_DIR / resource_name
            if not resource.exists():
                print(f"Error: Failed to load resource {resource}")
                exit(1)

            return pygame.image.load(resource)

        ## Load the pitch image and scale it to required size
        football_pitch_image = get_resource("football_pitch.png")
        self.football_pitch_image = pygame.transform.scale(
            football_pitch_image, (self.field_width, self.field_height)
        )

        ## Load the football image and scale it to required size
        football_image = get_resource("football.png")
        self.football_image = pygame.transform.scale(
            football_image, (2 * self.ball_size, 2 * self.ball_size)
        )

        ## Load the agent image and scale it to required size
        agent_image = get_resource("player.png")
        self.agent_image = pygame.transform.scale(
            agent_image, (2 * self.agent_size, 2 * self.agent_size)
        )

        ## Load the obstacle image and scale it to required size
        obstacle_image = get_resource("obstacle.png")
        self.obstacle_image = pygame.transform.scale(
            obstacle_image, (2 * self.obstacle_size, 2 * self.obstacle_size)
        )
