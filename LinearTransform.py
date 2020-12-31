class LinearTransform:
    def __init__(self, left, right, bottom, top, px_size, py_size, border):
        self.left    = left
        self.right   = right
        self.bottom  = bottom
        self.top     = top
        self.x_size  = right - left
        self.y_size  = top - bottom
        # --- Shift map in x or y direction ---
        self.pan_x  = 0
        self.pan_y  = 0

        print('LinearTransform() left       = {0}'.format(left))
        print('LinearTransform() right      = {0}'.format(right))
        print('LinearTransform() bottom     = {0}'.format(bottom))
        print('LinearTransform() top        = {0}'.format(top))
        print('LinearTransform() x_size     = {0}'.format(self.x_size))
        print('LinearTransform() y_size     = {0}'.format(self.y_size))

        # --- Calculate scale in [pixels] / [map_unit] ---
        self.px_size = px_size
        self.py_size = py_size
        self.border  = border
        self.border_x = px_size * border / 100
        self.border_y = py_size * border / 100
        self.pxsize_nob = px_size - 2*self.border_x
        self.pysize_nob = py_size - 2*self.border_y
        self.x_scale = self.pxsize_nob / float(self.x_size)
        self.y_scale = self.pysize_nob / float(self.y_size)
        if self.x_scale < self.y_scale:
            self.scale   = self.x_scale
            self.xoffset = self.border_x
            self.yoffset = (py_size - int(self.y_size*self.scale)) / 2
        else:
            self.scale   = self.y_scale
            self.xoffset = (px_size - int(self.x_size*self.scale)) / 2
            self.yoffset = self.border_y
        print('LinearTransform() px_size    = {0}'.format(px_size))
        print('LinearTransform() py_size    = {0}'.format(py_size))
        print('LinearTransform() border     = {0}'.format(border))
        print('LinearTransform() border_x   = {0}'.format(self.border_x))
        print('LinearTransform() border_y   = {0}'.format(self.border_y))
        print('LinearTransform() pxsize_nob = {0}'.format(self.pxsize_nob))
        print('LinearTransform() pysize_nob = {0}'.format(self.pysize_nob))
        print('LinearTransform() xscale     = {0}'.format(self.x_scale))
        print('LinearTransform() yscale     = {0}'.format(self.y_scale))
        print('LinearTransform() scale      = {0}'.format(self.scale))
        print('LinearTransform() xoffset    = {0}'.format(self.xoffset))
        print('LinearTransform() yoffset    = {0}'.format(self.yoffset))

    def MapToScreen(self, map_x, map_y):
        screen_x = self.scale * (+map_x - self.left) + self.xoffset
        screen_y = self.scale * (-map_y + self.top)  + self.yoffset

        return (int(screen_x), int(screen_y))

    def ScreenToMap(self, screen_x, screen_y):
        map_x = +(screen_x - self.xoffset + self.scale * self.left) / self.scale
        map_y = -(screen_y - self.yoffset - self.scale * self.top) / self.scale

        return (int(map_x), int(map_y))