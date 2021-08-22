import torch
import pyro
from pyro.nn import PyroModule, pyro_method
from pyro.distributions import Normal, TransformedDistribution
from pyro.distributions.torch_transform import ComposeTransformModule
from pyro.distributions.conditional import ConditionalTransformedDistribution
from pyro.distributions.transforms import (
    Spline, ExpTransform, ComposeTransform, ConditionalAffineCoupling,
    GeneralizedChannelPermute, SigmoidTransform
    )

from src.normalizingFlowsSCM.transforms import (
    ReshapeTransform, SqueezeTransform,
    TransposeTransform, LearnedAffineTransform,
    ConditionalAffineTransform, ActNorm
    )

from pyro.nn import DenseNN
from src.normalizingFlowsSCM.arch import BasicFlowConvNet


class FlowSCM(PyroModule):
    def __init__(self, use_affine_ex=True):
        super().__init__()
        self.num_scales = 2
        self.flows_per_scale = 1
        self.use_actnorm = False
        self.use_affine_ex = use_affine_ex

        self.register_buffer("glasses_base_loc", torch.zeros([1, ], requires_grad=False))
        self.register_buffer("glasses_base_scale", torch.ones([1, ], requires_grad=False))

        self.register_buffer("glasses_flow_lognorm_loc", torch.zeros([], requires_grad=False))
        self.register_buffer("glasses_flow_lognorm_scale", torch.ones([], requires_grad=False))

        self.glasses_flow_lognorm = AffineTransform(loc=self.glasses_flow_lognorm_loc.item(), scale=self.glasses_flow_lognorm_scale.item())

        self.glasses_flow_components = ComposeTransformModule([Spline(1)])
        self.glasses_flow_constraint_transforms = ComposeTransform([self.glasses_flow_lognorm,
            SigmoidTransform()])
        self.glasses_flow_transforms = ComposeTransform([self.glasses_flow_components,
            self.glasses_flow_constraint_transforms])

        glasses_base_dist = Normal(self.glasses_base_loc, self.glasses_base_scale).to_event(1)
        self.glasses_dist = TransformedDistribution(glasses_base_dist, self.glasses_flow_transforms)

        self._build_image_flow()
        self.register_buffer("x_base_loc", torch.zeros([1, 64, 64], requires_grad=False))
        self.register_buffer("x_base_scale", torch.ones([1, 64, 64], requires_grad=False))
        self.x_base_dist = Normal(self.x_base_loc, self.x_base_scale).to_event(3)

    def model(self):
        glasses_ = pyro.sample("glasses_", self.glasses_dist)
        glasses = pyro.sample("glasses", dist.Bernoulli(glasses_))
        glasses_context = self.glasses_flow_constraint_transforms.inv(glasses_)

        cond_x_transforms = ComposeTransform(
            ConditionalTransformedDistribution(self.x_base_dist, self.x_transforms)
            .condition(glasses_context).transforms
            ).inv
        cond_x_dist = TransformedDistribution(self.x_base_dist, cond_x_transforms)
        x = pyro.sample("x", cond_x_dist)

        return x, glasses


    def sample(self, n_samples=1):
        with pyro.plate("observations", n_samples):
            print(n_samples)
            samples = self.model()

        return (*samples,)


    def _build_image_flow(self):
        self.trans_modules = ComposeTransformModule([])
        self.x_transforms = []
        self.hidden_channels = 3

        c = 1
        for _ in range(self.num_scales):
            self.x_transforms.append(SqueezeTransform())
            c *= 4

            for _ in range(self.flows_per_scale):
                if self.use_actnorm:
                    actnorm = ActNorm(c)
                    self.trans_modules.append(actnorm)
                    self.x_transforms.append(actnorm)

                gcp = GeneralizedChannelPermute(channels=c)
                self.trans_modules.append(gcp)
                self.x_transforms.append(gcp)

                self.x_transforms.append(TransposeTransform(permutation=torch.tensor((1, 2, 0))))

                ac = ConditionalAffineCoupling(c // 2, BasicFlowConvNet(c // 2, self.hidden_channels, (c // 2, c // 2), 1))
                self.trans_modules.append(ac)
                self.x_transforms.append(ac)

                self.x_transforms.append(TransposeTransform(torch.tensor((2, 0, 1))))

            gcp = GeneralizedChannelPermute(channels=c)
            self.trans_modules.append(gcp)
            self.x_transforms.append(gcp)

        self.x_transforms += [
            ReshapeTransform((4**self.num_scales, 64 // 2**self.num_scales, 64 // 2**self.num_scales), (1, 64, 64))
        ]

        if self.use_affine_ex:
            affine_net = DenseNN(1, [16, 16], param_dims=[1, 1])
            affine_trans = ConditionalAffineTransform(context_nn=affine_net, event_dim=3)

            self.trans_modules.append(affine_trans)
            self.x_transforms.append(affine_trans)


