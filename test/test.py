import my_model
import nose

class testDirichlet(object):
    def setUp(self):
        self.dirichlet = my_model.Dirichlet(3.)

    def test(self):
        nose.tools.assert_equal(self.dirichlet.u_0, 3.)

# class testRobin(object):
#     def setUp(self):
#         self.Robin = my_model.
#     d = my_model.Dirichlet(4.)
#     nose.tools.assert_almost_equal(d.u_0, 4.)
