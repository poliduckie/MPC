# coding=utf-8
import numpy as np
from gym import spaces

from ..simulator import Simulator
from .. import logger


class DuckietownEnv(Simulator):
    """
    Wrapper to control the simulator using velocity and steering angle
    instead of differential drive motor velocities
    """

    def __init__(self, gain=1.0, trim=0.0, radius=0.0318, k=27.0, limit=1.0, **kwargs):
        Simulator.__init__(self, **kwargs)
        logger.info("using DuckietownEnv")

        self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)

        # Should be adjusted so that the effective speed of the robot is 0.2 m/s
        self.gain = gain

        # Directional trim adjustment
        self.trim = trim

        # Wheel radius
        self.radius = radius

        # Motor constant
        self.k = k

        # Wheel velocity limit
        self.limit = limit

    def step(self, action):
        vel, angle = action

        # Distance between the wheels
        baseline = self.unwrapped.wheel_dist

        # assuming same motor constants k for both motors
        k_r = self.k
        k_l = self.k

        # adjusting k by gain and trim
        k_r_inv = (self.gain + self.trim) / k_r
        k_l_inv = (self.gain - self.trim) / k_l

        omega_r = (vel + 0.5 * angle * baseline) / self.radius
        omega_l = (vel - 0.5 * angle * baseline) / self.radius

        # conversion from motor rotation rate to duty cycle
        u_r = omega_r * k_r_inv
        u_l = omega_l * k_l_inv

        # limiting output to limit, which is 1.0 for the duckiebot
        u_r_limited = max(min(u_r, self.limit), -self.limit)
        u_l_limited = max(min(u_l, self.limit), -self.limit)

        vels = np.array([u_l_limited, u_r_limited])

        obs, reward, done, info = Simulator.step(self, vels)
        mine = {}
        mine["k"] = self.k
        mine["gain"] = self.gain
        mine["train"] = self.trim
        mine["radius"] = self.radius
        mine["omega_r"] = omega_r
        mine["omega_l"] = omega_l
        info["DuckietownEnv"] = mine
        return obs, reward, done, info


class DuckietownLF(DuckietownEnv):
    """
    Environment for the Duckietown lane following task with
    and without obstacles (LF and LFV tasks)
    """

    def __init__(self, **kwargs):
        DuckietownEnv.__init__(self, **kwargs)

    def step(self, action):
        obs, reward, done, info = DuckietownEnv.step(self, action)
        return obs, reward, done, info


class DuckietownNav(DuckietownEnv):
    """
    Environment for the Duckietown navigation task (NAV)
    """

    def __init__(self, **kwargs):
        self.goal_tile = None
        DuckietownEnv.__init__(self, **kwargs)

    def reset(self, segment=False):
        DuckietownNav.reset(self)

        # Find the tile the agent starts on
        start_tile_pos = self.get_grid_coords(self.cur_pos)
        start_tile = self._get_tile(*start_tile_pos)

        # Select a random goal tile to navigate to
        assert len(self.drivable_tiles) > 1
        while True:
            tile_idx = self.np_random.randint(0, len(self.drivable_tiles))
            self.goal_tile = self.drivable_tiles[tile_idx]
            if self.goal_tile is not start_tile:
                break

    def step(self, action):
        obs, reward, done, info = DuckietownNav.step(self, action)

        info["goal_tile"] = self.goal_tile

        # TODO: add term to reward based on distance to goal?

        cur_tile_coords = self.get_grid_coords(self.cur_pos)
        cur_tile = self._get_tile(*cur_tile_coords)

        if cur_tile is self.goal_tile:
            done = True
            reward = 1000

        return obs, reward, done, info
'''
class lineD(DuckietownEnv):
    """
    Environment for the Duckietown lane following task with
    and without obstacles (LF and LFV tasks)
    """

    def __init__(self, **kwargs):
        DuckietownEnv.__init__(self, **kwargs)

        def draw_features(self):
            # @riza
            """This is for getting which part are we in tile-> right, or left"""
            # if self.step_count < 10:
            i, j = self.get_grid_coords(self.cur_pos)
            tile = self._get_tile(i, j)

            if tile is None or not tile['drivable']:
                return None

            curves = tile['curves']
            curve_headings = curves[:, -1, :] - curves[:, 0, :]
            curve_headings = curve_headings / np.linalg.norm(curve_headings).reshape(1, -1)
            dirVec = get_dir_vec(self.cur_angle)
            dot_prods = np.dot(curve_headings, dirVec)
            curve = np.argmax(dot_prods)

            # Curve points: 1->right, 0->left w.r.t car's perspective
            ii = curve # 1
            # Draw points on bezier curve on the current tile
            i, j = self.get_grid_coords(self.cur_pos)
            curves = self._get_tile(i, j)['curves']
            bezier_draw_points_curve(curves[ii])

            # draw points on the upcoming tile
            tile_coords = get_tiles(self.map_name)
            idx = tile_coords.index((i, j))
            # if we're at the end of the list return to beginning
            if len(tile_coords) - 1 == idx:
                i, j = tile_coords[0]
            else:
                i, j = tile_coords[idx + 1]

            curves_next = self._get_tile(i, j)['curves']
            bezier_draw_points_curve(curves_next[ii])
            # draw points on the previous tile
            i, j = tile_coords[idx - 1]
            curves_prev = self._get_tile(i, j)['curves']
            bezier_draw_points_curve(curves_prev[ii])

            # Get bezier points on 3 tiles(current, prev, next) and compute dist
            n = 50
            pts = np.vstack((
                [bezier_point(curves[ii], i / (n - 1)) for i in range(0, n)],
                [bezier_point(curves_next[ii], i / (n - 1)) for i in range(0, n)],
                [bezier_point(curves_prev[ii], i / (n - 1)) for i in range(0, n)]))

            bezier_draw_line(get_dir_line(self.cur_angle, self.cur_pos))
            # Draw the center of the robot
            draw_point(_actual_center(self.cur_pos, self.cur_angle))

            return compute_dist(get_dir_line(self.cur_angle, self.cur_pos),
                         list(pts),
                         get_dir_vec(self.cur_angle),#.self was wrong
                         debug=False)
                    
    def get_features(self):
        """
        Feature vector for DDPG
        """
        dists = np.array(self.draw_features()).flatten()
        # When using frame_skip, agent can get out of tile & bezier curve points can't be observed
        # hence, sensor readings will be None
        if None in dists:
            dists = np.zeros((1, 24))

        wheelVels = np.array([self.wheelVels[0], self.wheelVels[1]])
        speed = np.asarray(self.speed)
        # self.last_action
        # Get state representation
        state = np.concatenate((dists, wheelVels, speed), axis=None)
        # Concatenate last state & current state
        feature = np.concatenate((self.last_state, state), axis=None)
        # Store last state
        self.last_state = np.append(self.last_state, state)
        self.last_state = self.last_state[27:]
        assert len(self.last_state) == 189

        return feature

    def get_distance(self):
        """
        Distance from center of the robot to the closest curve point
        """
        # Get the actual center of car
        center = _actual_center(self.cur_pos, self.cur_angle)
        # Get closest curve point w.r.t. current position of car
        closest, _ = self.closest_curve_point(self.cur_pos, self.cur_angle)
        # Calculate distance b/w center and closest curve point
        dist = np.linalg.norm(center - closest)
        # Set z-dimension to 0.01 -> required for the line to be seen in the top-down view
        center[1], closest[1] = 0.01, 0.01
        # Draw the line
        bezier_draw_line(np.vstack((center, closest)))
        return dist

    def dist_centerline_curve(self):
        """
        Calculate the distance between the middle sensor line(center of robot actually) and the projected point
        on the curve.
        """
        DIST_NOT_INTERSECT = 5
        # Get directory line
        cps = get_dir_line(self.cur_angle, self.cur_pos)
        # Get the center point of directory line
        cps_center = (cps[0] + cps[1]) / 2
        # Get directory vector
        dir_vec = get_dir_vec(self.cur_angle)

        i, j = self.get_grid_coords(self.cur_pos)
        curves = self._get_tile(i, j)['curves']
        curve_headings = curves[:, -1, :] - curves[:, 0, :]
        curve_headings = curve_headings / np.linalg.norm(curve_headings).reshape(1, -1)
        dot_prods = np.dot(curve_headings, get_dir_vec(self.cur_angle))
        # Curve points: 1->right, 0->left w.r.t car's perspective
        ii = np.argmax(dot_prods)

        # draw points on the upcoming tile
        tile_coords = get_tiles(self.map_name)
        idx = tile_coords.index((i, j))
        # if we're at the end of the list return to beginning
        i, j = tile_coords[0] if len(tile_coords)-1 == idx else tile_coords[idx + 1]
        curves_next = self._get_tile(i, j)['curves']

        # draw points on the previous tile
        i, j = tile_coords[idx - 1]
        curves_prev = self._get_tile(i, j)['curves']

        # Get bezier points on 3 tiles(current, prev, next) and compute dist
        n = 50
        pts = np.vstack((
            [bezier_point(curves[ii], i / (n - 1)) for i in range(0, n)],
            [bezier_point(curves_next[ii], i / (n - 1)) for i in range(0, n)],
            [bezier_point(curves_prev[ii], i / (n - 1)) for i in range(0, n)]))

        # Calculate feature: whether there's an intersection & if so, the distance
        is_true, dist = compute_dist(np.vstack((cps_center, cps[1])), pts, dir_vec, n=1, red=True)[0]

        if is_true:
            print("DISTANCE FROM THE LINE: ", dist)
            return abs(dist)
        else:
            return DIST_NOT_INTERSECT#########################################end of riza'''


    
