import pygame


class Viewer:
	def __init__(self, options):
		pygame.init()

		self.field_width = options["field_width"]
		self.field_height = options["field_height"]
		self.agent_size = options["agent_size"]
		self.ball_size = options["ball_size"]

		self.screen = pygame.display.set_mode((self.field_width, self.field_height))
		self.clock = pygame.time.Clock()

		self._load_resources()

	def close(self):
		pygame.quit()


	def render(self, agent_positions, ball_position):
		self.screen.blit(self.football_pitch_image, (0, 0))

		for _, position in agent_positions.items():
			self.screen.blit(self.agent_image, position - (self.agent_image.get_width()/2))

		self.screen.blit(self.football_image, ball_position - (self.football_image.get_width()/2))
		pygame.display.flip()


	def _load_resources(self):
		football_pitch_image = pygame.image.load("resources/football_pitch.png")
		self.football_pitch_image = pygame.transform.scale(
			football_pitch_image, (self.field_width, self.field_height)
		)

		football_image = pygame.image.load("resources/football.png")
		self.football_image = pygame.transform.scale(
				football_image, (2 * self.ball_size, 2 * self.ball_size)
		)

		agent_image = pygame.image.load("resources/player.png")
		self.agent_image = pygame.transform.scale(
				agent_image, (2 * self.agent_size, 2 * self.agent_size)
		)
