import math

import cv2
import numpy as np
import plotly.graph_objects as go
from PIL import Image


PLOT_COLORs = {
    "border": "#E6770B",
    "tint_alpha": 0.0001,  # 左下角圆点右侧的线条的alpha（不能设为0）
    "affo": "#bc5090",
    "attr_neg": "#003F5C",
    "attr_pos": "#ffa600",
}

FILL_COLORs = {
    "border": PLOT_COLORs["border"] + "20",
    "affo": PLOT_COLORs["affo"],
    "attr_neg": PLOT_COLORs["attr_neg"],
    "attr_pos": PLOT_COLORs["attr_pos"],
}

CAUSAL_COLORs = {  # 右上角graph的颜色
    "positive": PLOT_COLORs["attr_pos"],
    "negative": PLOT_COLORs["attr_neg"],
    "default": PLOT_COLORs["affo"],
    "edge": "#D3290F",
}
EDGE_COLORs = {
    "positive": "#ff6361",
    "negative": "#58508d",
    "default": PLOT_COLORs["affo"],
    "edge": "#D3290F",
}


def prob2size(p):
    return 30 + 30 * p


def prob2color(is_positive):
    if is_positive:
        return CAUSAL_COLORs["positive"]
    else:
        return CAUSAL_COLORs["negative"]


def prob2edgecolor(is_positive):
    if is_positive:
        return EDGE_COLORs["positive"]
    else:
        return EDGE_COLORs["negative"]


def prob2sign(is_positive):
    if is_positive:
        return "+"
    else:
        return "-"


def prob2alpha(p, is_positive=False, activate_threshold=0.05):
    if p > activate_threshold:
        res = (p - activate_threshold) / (1 - activate_threshold)
        if res > 1:
            res = 1
        return res
    else:
        return 0

def hsv_to_rgb(h, s, v):
    """Convert HSV to RGB values (0-255)."""
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return int(r * 255), int(g * 255), int(b * 255)

def build_causal_graph_figure(
    logic_chains: list[dict],
    attr_prob_dict: dict[str, float],
    aff_prob_dict: dict[str, float],
    affordance_name: str,
):
    """Build a network-style causal graph with nodes and edges."""
    fig = go.Figure()
    
    # Chinese translation dictionary
    chinese_dict = {
        # Common attributes
        "plastic": "塑料",
        "metal": "金属",
        "wood": "木头",
        "glass": "玻璃",
        "ceramic": "陶瓷",
        "fabric": "布料",
        "rubber": "橡胶",
        "paper": "纸张",
        "leather": "皮革",
        "stone": "石头",
        "transparent": "透明",
        "opaque": "不透明",
        "smooth": "光滑",
        "rough": "粗糙",
        "soft": "柔软",
        "hard": "坚硬",
        "flexible": "柔韧",
        "rigid": "刚性",
        "light": "轻",
        "heavy": "重",
        "hot": "热",
        "cold": "冷",
        "wet": "湿",
        "dry": "干",
        "clean": "干净",
        "dirty": "脏",
        # Common affordances
        "operate": "可操作",
        "grasp": "抓取",
        "push": "推",
        "pull": "拉",
        "lift": "举起",
        "carry": "携带",
        "throw": "投掷",
        "squeeze": "挤压",
        "twist": "扭转",
        "open": "打开",
        "close": "关闭",
        "pour": "倾倒",
        "drink": "饮用",
        "eat": "食用",
        "cut": "切割",
        "break": "破坏",
        "bend": "弯曲",
        "fold": "折叠",
        "write": "书写",
        "read": "阅读",
        # State attributes
        "covered": "覆盖",
        "filled": "填充",
        "reversed": "反转",
        "empty": "空",
        "full": "满",
        "open": "开",
        "closed": "关",
    }
    
    # Extract all unique nodes from logic_chains
    attrs = set()
    affs = set()
    for chain in logic_chains:
        attrs.add(chain["attribute"])
        affs.add(chain["affordance"])
    attr_prob_dict["filled"] = 1.0
    # Combine all probability dictionaries
    all_probs = {**attr_prob_dict, **aff_prob_dict}
    # Create node positions (simple circular layout)
    import math
    all_nodes = list(attrs) + list(affs)
    n_nodes = len(all_nodes)
    
    node_positions = {}
    for i, node in enumerate(all_nodes):
        angle = 2 * math.pi * i / n_nodes
        x = 3 * math.cos(angle)
        y = 3 * math.sin(angle)
        node_positions[node] = (x, y)
    attr_is_pos = {}
    
    # Collect all existing connections from logic_chains
    existing_connections = set()
    for chain in logic_chains:
        attr = chain["attribute"]
        aff = chain["affordance"]
        attr_is_pos[attr] = chain["is_positive_affect"]
        # Store both directions to avoid duplicate lines
        existing_connections.add((attr, aff))
        existing_connections.add((aff, attr))
    
    # Draw all possible connections between nodes
    for i, node1 in enumerate(all_nodes):
        for j, node2 in enumerate(all_nodes):
            if i >= j:  # Avoid duplicate lines and self-connections
                continue
                
            pos1 = node_positions[node1]
            pos2 = node_positions[node2]
            
            # Check if this connection exists in logic_chains
            if (node1, node2) in existing_connections or (node2, node1) in existing_connections:
                # Find the corresponding logic chain for causal relationships
                causal_chain = None
                for chain in logic_chains:
                    if (chain["attribute"] == node1 and chain["affordance"] == node2) or \
                       (chain["attribute"] == node2 and chain["affordance"] == node1):
                        causal_chain = chain
                        break
                
                if causal_chain:
                    # This is a causal relationship
                    attr = causal_chain["attribute"]
                    is_positive = causal_chain["is_positive_affect"]
                    
                    # Calculate edge brightness based on node probabilities
                    attr_prob = attr_prob_dict.get(attr, 0.0)
                    edge_prob = attr_prob
                    opacity = 0.3 + 0.7 * min(edge_prob, 1.0)
                    
                    # Choose line style and color based on is_positive_affect
                    if is_positive:
                        line_style = dict(color=f"rgba(86, 86, 86, 0.3)", width=2, dash="dash")
                        hover_text = f"<b>{attr} → (无因果关系)</b><br>Edge Probability: {edge_prob:.3f}<extra></extra>"
                    else:
                        line_style = dict(color=f"rgba(244, 67, 54, {opacity})", width=2)
                        hover_text = f"<b>{attr} → (因果关系)</b><br>Edge Probability: {edge_prob:.3f}<br>Opacity: {opacity:.3f}<extra></extra>"
                else:
                    # Fallback for existing connections without clear causal data
                    line_style = dict(color=f"rgba(86, 86, 86, 0.3)", width=2, dash="dash")
                    hover_text = f"<b>{node1} ↔ {node2}</b><br>无因果关系<extra></extra>"
            else:
                # No causal relationship - draw dashed gray line
                line_style = dict(color=f"rgba(86, 86, 86, 0.3)", width=2, dash="dash")
                hover_text = f"<b>{node1} ↔ {node2}</b><br>无因果关系<extra></extra>"
            
            fig.add_trace(
                go.Scatter(
                    x=[pos1[0], pos2[0]],
                    y=[pos1[1], pos2[1]],
                    mode="lines",
                    line=line_style,
                    showlegend=False,
                    hovertemplate=hover_text,
                )
            )
    
    # Draw nodes with brightness based on probability values
    for node in all_nodes:
        pos = node_positions[node]
        prob = all_probs.get(node, 0.0)
        
        # Calculate brightness using HSV (0.3 to 1.0 range for value)
        brightness = 1.0 # 0.3 + 0.7 * min(prob, 1.0)
        # Different base colors for attrs vs affs using HSV
        if node in attrs and not attr_is_pos[node]:
            # Blue hue for attributes (hue=0.6, saturation=0.8)
            r, g, b = hsv_to_rgb(0.6, 0.8, brightness)
            a = 1.0
            base_size = 40 + 80 * min(prob, 1.0)
        else:
            # Orange hue for affordances (hue=0.1, saturation=0.8)
            r, g, b = 51, 133, 255
            a = 0.3
            # Calculate size based on probability (40 to 80 range)
            base_size = 40
        
        # Highlight the selected affordance
        if node == affordance_name:
            # Pink hue for selected affordance (hue=0.9, saturation=0.8)
            r, g, b = hsv_to_rgb(0.9, 0.8, brightness)
            a = 1.0
            size = base_size + 10  # Slightly larger for selected affordance
        else:
            size = base_size
        
        color = f"rgb({r}, {g}, {b}, {a})"
        
        # Get Chinese translation
        chinese_text = chinese_dict.get(node, node)
        
        fig.add_trace(
            go.Scatter(
                x=[pos[0]],
                y=[pos[1]],
                mode="markers+text",
                marker=dict(size=size, color=color, line=dict(width=2, color="white")),
                text=chinese_text,
                textposition="middle center",
                textfont=dict(size=10, color="white", family="Arial Black"),
                showlegend=False,
                hovertemplate=f"<b>{chinese_text}</b> ({node})<br>Probability: {prob:.3f}<extra></extra>",
            )
        )
    
    # Add legend
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line=dict(color="#565656", width=3, dash="dash"),
            name="无因果关系",
            showlegend=True,
        )
    )
    # fig.add_trace(
    #     go.Scatter(
    #         x=[None], y=[None],
    #         mode="lines", 
    #         line=dict(color="#F44336", width=3),
    #         name="有因果关系影响",
    #         showlegend=True,
    #     )
    # )
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line=dict(color="rgba(244, 67, 54, 0.1)", width=3),
            name="因果关系小",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line=dict(color="rgba(244, 67, 54, 1.0)", width=3),
            name="因果关系大",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=20, color="rgba(33, 150, 243, 0.8)"),
            name="属性",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=20, color="rgba(255, 152, 0, 0.8)"),
            name="可供性",
            showlegend=True,
        )
    )
    
    # Configure layout
    fig.update_layout(
        showlegend=True,
        legend=dict(x=0, y=1, bgcolor="rgba(255, 255, 255, 0.8)"),
        xaxis=dict(visible=False, range=[-4, 4]),
        yaxis=dict(visible=False, range=[-4, 4], scaleanchor="x"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=20, r=20, t=20, b=20),
        height=400,
    )
    
    return fig
    
def build_causal_graph_figure_bak(
    logic_chains: list[dict],
    attr_prob_dict: dict[str, float],
    aff_prob_dict: dict[str, float],
    affordance_name: str,
):
    """Build and return a Plotly Figure for the causal graph view."""
    related_edges = [
        edge for edge in logic_chains if edge["affordance"] == affordance_name
    ]
    pos_attribute_names = [
        edge["attribute"]
        for edge in related_edges
        if edge["is_positive_affect"]
    ]
    neg_attribute_names = [
        edge["attribute"]
        for edge in related_edges
        if not edge["is_positive_affect"]
    ]

    fig = go.Figure()
    ATTRIBUTE_UPPERBOUND = 8.6  # const variable
    ATTRIBUTE_LOWBOUND = 5
    AFFORDANCE_UPPERBOUND = 180
    AFFORDANCE_LOWBOUND = 20

    pos_attributes = {
        name: attr_prob_dict[name] for name in pos_attribute_names
    }
    neg_attributes = {
        name: attr_prob_dict[name] for name in neg_attribute_names
    }
    affordance_value = {affordance_name: aff_prob_dict[affordance_name]}

    pos_attribute_size = {
        name: max(
            ATTRIBUTE_LOWBOUND,
            min(math.log(value + 1) * 25, ATTRIBUTE_UPPERBOUND),
        )
        for name, value in pos_attributes.items()
    }
    neg_attribute_size = {
        name: max(
            ATTRIBUTE_LOWBOUND,
            min(math.log(value + 1) * 25, ATTRIBUTE_UPPERBOUND),
        )
        for name, value in neg_attributes.items()
    }
    size = max(
        AFFORDANCE_LOWBOUND,
        min(
            math.log(affordance_value[affordance_name] + 1) * 50,
            AFFORDANCE_UPPERBOUND,
        ),
    )

    avg_pos = (
        sum(pos_attributes.values()) / len(pos_attributes)
        if len(pos_attributes) > 0
        else 0
    )
    avg_neg = (
        sum(neg_attributes.values()) / len(neg_attributes)
        if len(neg_attributes) > 0
        else 0
    )

    y_aff = (
        0.9 * avg_pos
        - 0.9 * avg_neg
        + 1.1 * sum(affordance_value.values())
    ) * 5
    prob = max(0, min(100, (y_aff + 2) * 10))
    y_neg = -5
    x_mid_pos = 0

    y_pos = 5 + y_aff

    if y_aff > 0.2:
        y_neg = -5 + (y_aff - 0.2)

    # affordance气球
    fig.add_trace(
        go.Scatter(
            x=[x_mid_pos],
            y=[y_aff],
            mode="markers+text",
            marker=dict(size=24, color="#EFB6C8", symbol="square"),
            text=affordance_name + f" (able):{prob:.2f}%",
            textposition="middle right",
            textfont=dict(color="black", size=18),
            showlegend=False,
            zorder=10,
        ),
    )

    # 加减号
    if pos_attribute_names:
        if len(pos_attribute_names) % 2 == 0:
            fig.add_trace(
                go.Scatter(
                    x=[x_mid_pos],
                    y=(
                        [(y_pos + y_aff + 1) / 2]
                        if (y_pos + y_aff + 1) / 2 < 8.5
                        else [8.5]
                    ),
                    mode="text",
                    text=["+"],
                    textposition="middle center",
                    textfont=dict(color="#FFB200", size=15, weight="bold"),
                    showlegend=False,
                ),
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=[x_mid_pos + 0.03],
                    y=(
                        [(y_pos + y_aff + 1) / 2]
                        if (y_pos + y_aff + 1) / 2 < 8.5
                        else [8.5]
                    ),
                    mode="text",
                    text=["+"],
                    textposition="middle right",
                    textfont=dict(color="#FFB200", size=15, weight="bold"),
                    showlegend=False,
                ),
            )
    if neg_attribute_names:
        if len(neg_attribute_names) % 2 == 0:
            fig.add_trace(
                go.Scatter(
                    x=[x_mid_pos],
                    y=(
                        [(y_neg + y_aff - 1) / 2]
                        if (y_neg + y_aff - 1) / 2 > -8.5
                        else [-8.5]
                    ),
                    mode="text",
                    text=["-"],
                    textposition="middle center",
                    textfont=dict(color="#EB5B00", size=15, weight="bold"),
                    showlegend=False,
                ),
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=[x_mid_pos + 0.04],
                    y=(
                        [(y_neg + y_aff - 1) / 2]
                        if (y_neg + y_aff - 1) / 2 > -8.5
                        else [-8.5]
                    ),
                    mode="text",
                    text=["-"],
                    textposition="middle right",
                    textfont=dict(color="#EB5B00", size=15, weight="bold"),
                    showlegend=False,
                ),
            )

    x_pos = -(len(pos_attributes) - 1) * 0.3 / 2
    balloon_image_path = "resources/balloon.png"
    balloon_image = Image.open(balloon_image_path)
    for attr in pos_attributes.keys():
        if y_pos + (pos_attribute_size[attr] - 5) * 0.54 > 9:
            y_pos_edited = 9 - (pos_attribute_size[attr] - 5) * 0.54
        else:
            y_pos_edited = y_pos
        fig.add_trace(
            go.Scatter(
                x=[x_pos, x_mid_pos],
                y=[y_pos_edited, y_aff],
                mode="lines",
                line=dict(color="#FFB200", width=2, dash="dash"),
                showlegend=False,
                zorder=2,
            ),
        )
        fig.add_layout_image(
            dict(
                source=balloon_image,
                xref="x",
                yref="y",
                x=x_pos,
                y=y_pos_edited + (pos_attribute_size[attr] - 5) * 0.27,
                sizex=pos_attribute_size[attr] * 1.1,
                sizey=pos_attribute_size[attr] * 1.1,
                xanchor="center",
                yanchor="middle",
            ),
        )
        fig.add_trace(
            go.Scatter(
                x=[x_pos],
                y=[
                    y_pos_edited
                    + (pos_attribute_size[attr] - 5) * 0.54
                    + 1.5
                ],
                mode="text",
                text= attr,
                textposition="top center",
                textfont=dict(color="black", size=18),
                showlegend=False,
            ),
        )
        fig.add_trace(
            go.Scatter(
                x=[x_pos, (x_pos + x_mid_pos) / 2, x_mid_pos],
                y=[y_pos_edited, (y_pos_edited + y_aff) / 2, y_aff],
                mode="markers",
                marker=dict(
                    symbol="arrow-down",
                    size=6,
                    color="#FFB200",
                    angleref="previous",
                ),
                showlegend=False,
                zorder=2,
            ),
        )
        x_pos += 0.3

    # arrow lines and balloons of neg attr
    x_neg = -(len(neg_attributes) - 1) * 0.3 / 2
    weight_image_path = "resources/weight.png"
    weight_image = Image.open(weight_image_path)
    neg_line_width = (y_aff - y_neg) / 3
    for attr in neg_attributes.keys():
        fig.add_trace(
            go.Scatter(
                x=[x_neg, x_mid_pos],
                y=[y_neg, y_aff],
                mode="lines",
                line=dict(color="#EB5B00", width=neg_line_width),
                showlegend=False,
                zorder=2,
            ),
        )
        fig.add_layout_image(
            dict(
                source=weight_image,
                xref="x",
                yref="y",
                x=x_neg,
                y=y_neg + (neg_attribute_size[attr] - 5) / 5.5,
                sizex=neg_attribute_size[attr] / 2.5,
                sizey=neg_attribute_size[attr] / 2.5,
                xanchor="center",
                yanchor="middle",
            ),
        )
        fig.add_trace(
            go.Scatter(
                x=[x_neg],
                y=[
                    y_neg
                    + (neg_attribute_size[attr] - 5) / 5.5
                    - neg_attribute_size[attr] / 5
                ],
                mode="text",
                text=attr,
                textposition="bottom center",
                textfont=dict(color="black", size=18),
                showlegend=False,
            ),
        )
        fig.add_trace(
            go.Scatter(
                x=[x_neg, (x_neg + x_mid_pos) / 2, x_mid_pos],
                y=[y_neg, (y_neg + y_aff) / 2, y_aff],
                mode="markers",
                marker=dict(
                    symbol="arrow-down",
                    size=6,
                    color="#EB5B00",
                    angleref="previous",
                ),
                showlegend=False,
                zorder=2,
            ),
        )
        x_neg += 0.3

    # Add custom legend using line styles and shapes
    # No background box needed
    
    # Legend items
    # 1. Dashed line - Positive attributes
    fig.add_trace(
        go.Scatter(
            x=[-0.92, -0.78],
            y=[10.5, 10.5],
            mode="lines",
            line=dict(color="#FFB200", width=3, dash="dash"),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_annotation(
        x=-0.7,
        y=10.5,
        text="Positive Attributes",
        showarrow=False,
        xanchor="left",
        yanchor="middle",
        font=dict(size=10, color="black"),
    )
    
    # 2. Solid line - Negative attributes
    fig.add_trace(
        go.Scatter(
            x=[-0.92, -0.78],
            y=[10.0, 10.0],
            mode="lines",
            line=dict(color="#EB5B00", width=3),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_annotation(
        x=-0.7,
        y=10.0,
        text="Negative Attributes",
        showarrow=False,
        xanchor="left",
        yanchor="middle",
        font=dict(size=10, color="black"),
    )
    
    # 3. Pink square - Affordance
    fig.add_trace(
        go.Scatter(
            x=[-0.85],
            y=[9.5],
            mode="markers",
            marker=dict(size=12, color="#EFB6C8", symbol="square"),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_annotation(
        x=-0.7,
        y=9.5,
        text="Affordance",
        showarrow=False,
        xanchor="left",
        yanchor="middle",
        font=dict(size=10, color="black"),
    )
    
    fig.update_xaxes(range=[-1, 1])
    fig.update_yaxes(range=[-11.5, 11.5])
    fig.update_layout(
        xaxis=dict(
            visible=False,
            showgrid=False,
            showticklabels=False,
            zeroline=False,
        ),
        yaxis=dict(
            visible=False,
            showgrid=False,
            showticklabels=False,
            zeroline=False,
        ),
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig


def draw_causal_graph(
    logic_chains: list[dict],
    attr_prob_dict: dict[str, float],
    aff_prob_dict: dict[str, float],
    affordance_name: str,
):
    """Backward-compatible helper that returns a PNG image (bytes).

    Internally builds the figure then exports to a cropped PNG, keeping the
    original behavior used by the OpenCV path.
    """
    fig = build_causal_graph_figure(
        logic_chains=logic_chains,
        attr_prob_dict=attr_prob_dict,
        aff_prob_dict=aff_prob_dict,
        affordance_name=affordance_name,
    )

    # Convert Plotly figure to image array with higher resolution
    img = fig.to_image(format="png", scale=3)
    img_array = np.frombuffer(img, np.uint8)
    img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Crop the image to half its width, centered
    height, width, _ = img_cv.shape
    start_x = width // 5
    start_y = height // 15
    end_y = height - int(height / 3.8)
    end_x = start_x + (width // 5) * 3
    cropped_img = img_cv[start_y:end_y, start_x:end_x]

    # Encode the cropped image back to bytes
    _, img_encoded = cv2.imencode(".png", cropped_img)
    return img_encoded.tobytes()
