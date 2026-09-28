*Script outline*
1. *Intro + Context*
	1. I've been playing this game - *small display of Idleon* - for over 5 years, but there's one part I just really suck at. 
	2. But what I don't suck at is programming, and if there's one thing I'm particularly good at, it's using the wrong tool for the right job.
	3. So I'm going to stuff a square peg on a round hole and use coding and AI to beat a silly minigame, because what I lack in patience, I make up for in laziness.
2. *What is Idleon?* 
	1. That game we'll be tackling is Idleon, specifically one of its minigames, hoop shooting
	2. Idleon is all about a bunch of connected skills and features that give stat bonuses. Alchemy lends stats for trapping, which lends stats for dreams, etcetera etcetera.
	3. Hoops gives quite a good trophy - and stat bonuses - if you reach far enough into it, and that's exactly what were going for
3. *The game plan*
	1. So, the proposed plan is this
	2. I'm gonna put my very expensive engineering degree to good use, and use AI to play Hoops optimally.
	3. To do that, first we need to first break up the rules of the game, and figure out the mechanics
	4. Secondly, well need to find a way to turn that info into something AI can use, and turn that into gameplay,
	5. And thirdly, test and iterate if it all works.
4. *The rules of the game - Phase 1*
	1. So, here's the game - *screenshot of game* - and a couple of things we need to take note of
	2. We get a point each time we put the basketball through the hoop. Simple.
	3. In the first phase of the game, the player moves vertically
	4. Secondly, the hoop spawns pretty much anywhere on the right hand side of the screen.
5.  *Breaking down Phase 1*
	1. We have two pieces of information we care about here, and while it looks complex, it's not that bad.
	2. We first need to know where the player is. We'll do that by tracking this cropped picture of the ball.
	3. Secondly, we need to know where the hoop is. For that, I'm using the back panel of the hoop.
	4. So we have 4 input:
		1. The players x and y position, and
		2. The hoops x and y
	5. Normally we wouldn't care about non changing values, such as the players x position, but we'll use it for phase 2.
	6. For AI, it's sometimes a good idea to turn unrelated data into relative data. In this case, that works by using the difference between two values.
	7. Using this, we can make 4 inputs 2.
6. *A quick note on the AI were using
	1. I'll be using Proximal Policy Optimization, or PPO for our training. 
	2. I've picked it for a few reasons, but mainly because it's simple to implement, and I've got a little bit of experience with it.
	3. To explain it briefly, it learns by a reward function. In our case, im going to punish it for missing, with a worse punishment the further it misses, and reward it for a score.
7. *How we're finding the ball*
	1. The other question is, how do we find our ball and hoop?
	2. For that, we'll be using OpenCV2, an image detection library with some pretty powerful features. I'll be building everything in Python.
	3. So let's give that a test and see how far we've come. *Show training here*
8. *The rules of the game - Phase 2
	1. Now we're easily getting our first 10 points, but that this stage the rules change.
	2. For the 2nd phase, the hoop starts moving in a horizontal fashion.
	3. We'll need to know some information on where the hoop is ***going to be*** to actually hit it.
	4. After some testing and many failed runs, the hoop always completes a cycle in 4 seconds. 
	5. I'm going to implement a fast way to detect where the hoop is, using something called the nyquist frequency.
	6. In simple terms, we can accurately rebuild a signal - in this case the horizontal movement - using a sampling rate half of the frequency or higher. I'm lazy, so we pick 0.25s, for 4hz, which is 16x higher than the frequency of the movement.
	7. Once we have the real movement, we can add more input to our AI. In this case, instead of handing it all the information about the movement such as speed, rate of change, etc. We instead simplify and hand it the distance as point of impact if we were to shoot right now. That's a linear equation we can skip for now, but trust me it will work.
9. *Rules of the game - Phase 3*
	1. Nice, were hitting 20 points now. At this stage, we hit phase 3 of the game, with one more rule change. The player is now also moving horizontally.
	2. This, to a normal player, seems really weird to play with, but we're actually already accounting for that. Since the ball is played from the player's immediate position, we don't need to account for time.
	3. We can simply make sure our distance to the hoop is presented accurately, and we should be done.
	4. Let's see it in action, I'll speed it up a little.
10. *The sign off*
	1. To recap, we used AI, PPO specificlly, to train a bot to throw hoops. Using it, I was able to get the 40 score needed for the trophy in little under XXX shots, or roughly XXX minutes, much better than the hundreds I've used to score in the past.
	2. If you found this entertaining or educational, consider subscribing. I want to tackle some more of Idleon's minigames, so let me know which you'd want to see next, between the mining mini game, bug catching, darts, or fishing. Cheers!