import json
from at_element import ATElement, ATElementType
import math

class ATConfiguration:
    def __init__(self):
        self.alpha_l = 2.0
        self.alpha_t = 0.2
        self.beta = 1 / (2 * self.alpha_l)
        self.ca = 8.0
        self.gamma = 3.5
        self.dom_xmin = 0.0
        self.dom_xmax = 600.0
        self.dom_ymin = -10.0
        self.dom_ymax = 40.0
        self.dom_inc = 0.5
        self.num_cp = 100
        self.num_terms = 7
        self.elements = []
        self.orientation = "horizontal"
        self.plot_aspect = "scaled"

    @staticmethod
    def from_json(filename):
        with open(filename, 'r') as f:
            data = json.load(f)

        config = ATConfiguration()
        config.alpha_l = data.get('alpha_l', config.alpha_l)
        config.alpha_t = data.get('alpha_t', config.alpha_t)
        config.beta = 1 / (2 * config.alpha_l)
        config.ca = data.get('ca', config.ca)
        config.gamma = data.get('gamma', config.gamma)
        config.dom_xmin = data.get('dom_xmin', config.dom_xmin)
        config.dom_xmax = data.get('dom_xmax', config.dom_xmax)
        config.dom_ymin = data.get('dom_ymin', config.dom_ymin)
        config.dom_ymax = data.get('dom_ymax', config.dom_ymax)
        config.dom_inc = data.get('dom_inc', config.dom_inc)
        config.num_cp = data.get('num_cp', config.num_cp)
        config.num_terms = data.get('num_terms', config.num_terms)
        config.orientation = data.get('orientation', config.orientation)
        config.plot_aspect = data.get('plot_aspect', config.plot_aspect)

        for i, elem_data in enumerate(data.get('elements', [])):
            kind = elem_data['kind'].lower()
            if kind == 'circle':
                elem = ATElement(
                    kind=ATElementType.Circle,
                    x=elem_data['x'],
                    y=elem_data['y'],
                    c=elem_data['c'],
                    r=elem_data['r']
                )
            elif kind == 'line':
                theta_deg = elem_data.get('theta',90)
                elem = ATElement(
                    kind=ATElementType.Line,
                    x=elem_data['x'],
                    y=elem_data['y'],
                    c=elem_data['c'],
                    r=elem_data['l']/2.0,  #radius
                    theta=math.radians(theta_deg)  # default vertical
                )
            else:
                raise ValueError(f"Unknown element kind '{elem_data['kind']}'")

            elem.label = elem_data.get('label', f"Element {i + 1}")
            elem.id = elem_data.get('id', f"source_{len(config.elements)}")
            config.elements.append(elem)
        return config

    def annotate_elements_on_plot(self, ax):
        for elem in self.elements:
            ax.text(elem.x, elem.y, elem.label, fontsize=14, ha='center', va='center', color='black', bbox=dict(facecolor='white', alpha=0.5))
