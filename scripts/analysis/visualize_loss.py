#!/usr/bin/env python
"""
Visualize loss implementation from train_multi.py
Shows how each loss component behaves with respect to time t
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Set style with better spacing
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10

# Time range
t = np.linspace(0, 1, 1000)

# ==================== Figure 1: Main Loss Visualization ====================
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.4, 
              left=0.08, right=0.95, top=0.95, bottom=0.08)

# ==================== 1. Velocity Loss ====================
ax1 = fig.add_subplot(gs[0, 0])
velocity_loss_base = 0.5
velocity_loss = np.full_like(t, velocity_loss_base)
ax1.plot(t, velocity_loss, 'b-', linewidth=2.5, label='Velocity Loss (MSE)', zorder=3)
ax1.fill_between(t, 0, velocity_loss, alpha=0.25, color='blue', zorder=1)
ax1.set_xlabel('Time t', fontsize=12, fontweight='bold')
ax1.set_ylabel('Loss Value', fontsize=12, fontweight='bold')
ax1.set_title('Velocity Loss (Flow Matching)', fontsize=13, fontweight='bold', pad=12)
ax1.legend(fontsize=9, loc='upper right', framealpha=0.95, frameon=True)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xlim(-0.02, 1.02)
ax1.set_ylim(0, velocity_loss_base * 1.2)
# Move text box to avoid legend overlap
ax1.text(0.5, 0.75, 'MSE(predicted_velocity,\ntrue_velocity)', 
         transform=ax1.transAxes, ha='center', fontsize=9, 
         bbox=dict(boxstyle='round,pad=0.6', facecolor='lightblue', 
                  edgecolor='blue', linewidth=1.5, alpha=0.9))

# ==================== 2. Distance Geometry Loss ====================
ax2 = fig.add_subplot(gs[0, 1])
max_dg_weight = 0.1
dg_weight = t * max_dg_weight
violation_amount = 0.5
dg_loss_unweighted = violation_amount
dg_loss = dg_loss_unweighted * dg_weight

ax2.plot(t, dg_loss, 'g-', linewidth=2.5, label='DG Loss (weighted)', zorder=3)
ax2.plot(t, dg_weight, 'g--', linewidth=1.8, alpha=0.6, label='Time Weight', zorder=2)
ax2.fill_between(t, 0, dg_loss, alpha=0.25, color='green', zorder=1)
ax2.set_xlabel('Time t', fontsize=12, fontweight='bold')
ax2.set_ylabel('Loss Value', fontsize=12, fontweight='bold')
ax2.set_title('Distance Geometry Loss', fontsize=13, fontweight='bold', pad=12)
ax2.legend(fontsize=9, loc='upper left', framealpha=0.95, frameon=True)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xlim(-0.02, 1.02)
ax2.set_ylim(0, max(dg_loss) * 1.2)
# Move text box to avoid legend overlap
ax2.text(0.5, 0.75, 'weight = t × 0.1\n(t=1 → weight=0.1)\nReLU(bound - dist)²', 
         transform=ax2.transAxes, ha='center', fontsize=9,
         bbox=dict(boxstyle='round,pad=0.6', facecolor='lightgreen', 
                  edgecolor='green', linewidth=1.5, alpha=0.9))

# ==================== 3. Clash Loss (CA) ====================
ax3 = fig.add_subplot(gs[1, 0])
ca_threshold = 3.0
clash_margin = 1.0
max_clash_weight = 0.1
clash_weight = t * max_clash_weight
clash_violation = 0.3
ca_clash_penalty = clash_violation ** 2
ca_clash_loss = ca_clash_penalty * clash_weight

ax3.plot(t, ca_clash_loss, 'r-', linewidth=2.5, label='CA Clash Loss', zorder=3)
ax3.plot(t, clash_weight, 'r--', linewidth=1.8, alpha=0.6, label='Time Weight', zorder=2)
ax3.fill_between(t, 0, ca_clash_loss, alpha=0.25, color='red', zorder=1)
ax3.set_xlabel('Time t', fontsize=12, fontweight='bold')
ax3.set_ylabel('Loss Value', fontsize=12, fontweight='bold')
ax3.set_title(f'CA Clash Loss (threshold={ca_threshold}Å)', fontsize=13, fontweight='bold', pad=12)
ax3.legend(fontsize=9, loc='upper left', framealpha=0.95, frameon=True)
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.set_xlim(-0.02, 1.02)
ax3.set_ylim(0, max(ca_clash_loss) * 1.2)
# Move text box to avoid legend overlap
ax3.text(0.5, 0.75, f'ReLU({ca_threshold} + {clash_margin} - dist)²\n× (t × 0.1)', 
         transform=ax3.transAxes, ha='center', fontsize=9,
         bbox=dict(boxstyle='round,pad=0.6', facecolor='mistyrose', 
                  edgecolor='red', linewidth=1.5, alpha=0.9))

# ==================== 4. Clash Loss (SC) ====================
ax4 = fig.add_subplot(gs[1, 1])
sc_threshold = 2.5
sc_clash_penalty = clash_violation ** 2
sc_clash_loss = sc_clash_penalty * clash_weight

ax4.plot(t, sc_clash_loss, color='#FF8C00', linewidth=2.5, label='SC Clash Loss', zorder=3)
ax4.plot(t, clash_weight, color='#FF8C00', linestyle='--', linewidth=1.8, 
         alpha=0.6, label='Time Weight', zorder=2)
ax4.fill_between(t, 0, sc_clash_loss, alpha=0.25, color='#FF8C00', zorder=1)
ax4.set_xlabel('Time t', fontsize=12, fontweight='bold')
ax4.set_ylabel('Loss Value', fontsize=12, fontweight='bold')
ax4.set_title(f'SC Clash Loss (threshold={sc_threshold}Å)', fontsize=13, fontweight='bold', pad=12)
ax4.legend(fontsize=9, loc='upper left', framealpha=0.95, frameon=True)
ax4.grid(True, alpha=0.3, linestyle='--')
ax4.set_xlim(-0.02, 1.02)
ax4.set_ylim(0, max(sc_clash_loss) * 1.2)
# Move text box to avoid legend overlap
ax4.text(0.5, 0.75, f'ReLU({sc_threshold} + {clash_margin} - dist)²\n× (t × 0.1)', 
         transform=ax4.transAxes, ha='center', fontsize=9,
         bbox=dict(boxstyle='round,pad=0.6', facecolor='wheat', 
                  edgecolor='#FF8C00', linewidth=1.5, alpha=0.9))

# ==================== 5. Total Loss ====================
ax5 = fig.add_subplot(gs[2, :])
total_loss = velocity_loss + dg_loss + ca_clash_loss + sc_clash_loss

# Stacked area plot
ax5.fill_between(t, 0, velocity_loss, alpha=0.3, color='blue', label='Velocity Loss', zorder=1)
ax5.fill_between(t, velocity_loss, velocity_loss + dg_loss, alpha=0.3, color='green', 
                 label='DG Loss', zorder=1)
ax5.fill_between(t, velocity_loss + dg_loss, velocity_loss + dg_loss + ca_clash_loss, 
                 alpha=0.3, color='red', label='CA Clash Loss', zorder=1)
ax5.fill_between(t, velocity_loss + dg_loss + ca_clash_loss, total_loss, 
                 alpha=0.3, color='#FF8C00', label='SC Clash Loss', zorder=1)

# Line plots on top
ax5.plot(t, velocity_loss, 'b-', linewidth=2, alpha=0.9, zorder=2)
ax5.plot(t, velocity_loss + dg_loss, 'g-', linewidth=2, alpha=0.9, zorder=2)
ax5.plot(t, velocity_loss + dg_loss + ca_clash_loss, 'r-', linewidth=2, alpha=0.9, zorder=2)
ax5.plot(t, total_loss, 'k-', linewidth=3, label='Total Loss', alpha=0.95, zorder=3)

ax5.set_xlabel('Time t', fontsize=12, fontweight='bold')
ax5.set_ylabel('Loss Value', fontsize=12, fontweight='bold')
ax5.set_title('Total Loss Composition', fontsize=13, fontweight='bold', pad=12)
# Move legend to top right to avoid overlap
ax5.legend(fontsize=9, loc='upper left', framealpha=0.95, ncol=5, columnspacing=1.0)
ax5.grid(True, alpha=0.3, linestyle='--')
ax5.set_xlim(-0.02, 1.02)
ax5.set_ylim(0, max(total_loss) * 1.05)

# Move formula text higher to avoid overlap
formula_text = (
    'Total Loss = Velocity Loss + DG Loss + CA Clash Loss + SC Clash Loss  |  '
    'Constraint losses are time-weighted: weight(t) = t × max_weight'
)
ax5.text(0.5, 0.05, formula_text, transform=ax5.transAxes, ha='center', 
         fontsize=9, bbox=dict(boxstyle='round,pad=0.6', facecolor='lightyellow', 
                              edgecolor='gray', linewidth=1, alpha=0.95))

plt.suptitle('Loss Implementation Visualization (train_multi.py)', 
             fontsize=15, fontweight='bold', y=0.98)

plt.savefig('loss_visualization.png', dpi=150, bbox_inches='tight', facecolor='white')
print("✓ Saved loss_visualization.png")

# ==================== Figure 2: Loss Components Breakdown ====================
fig2, axes = plt.subplots(1, 3, figsize=(17, 5.5))
fig2.patch.set_facecolor('white')

# Pie chart
t_final = 1.0
v_loss_final = velocity_loss_base
dg_loss_final = dg_loss_unweighted * max_dg_weight * t_final
ca_loss_final = ca_clash_penalty * max_clash_weight * t_final
sc_loss_final = sc_clash_penalty * max_clash_weight * t_final

losses_at_t1 = [v_loss_final, dg_loss_final, ca_loss_final, sc_loss_final]
labels = ['Velocity', 'DG', 'CA Clash', 'SC Clash']
colors = ['#4169E1', '#32CD32', '#DC143C', '#FF8C00']
explode = (0.05, 0.05, 0.05, 0.05)

wedges, texts, autotexts = axes[0].pie(losses_at_t1, labels=labels, colors=colors, 
                                       autopct='%1.1f%%', startangle=90, explode=explode, 
                                       shadow=True, textprops={'fontsize': 10, 'fontweight': 'bold'})
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
axes[0].set_title('Loss Components at t=1.0', fontsize=12, fontweight='bold', pad=18)

# Bar chart
time_points = [0.0, 0.25, 0.5, 0.75, 1.0]
v_vals = [velocity_loss_base] * len(time_points)
dg_vals = [dg_loss_unweighted * max_dg_weight * tp for tp in time_points]
ca_vals = [ca_clash_penalty * max_clash_weight * tp for tp in time_points]
sc_vals = [sc_clash_penalty * max_clash_weight * tp for tp in time_points]
total_vals = [v + dg + ca + sc for v, dg, ca, sc in zip(v_vals, dg_vals, ca_vals, sc_vals)]

x = np.arange(len(time_points))
width = 0.16
axes[1].bar(x - 1.5*width, v_vals, width, label='Velocity', color='#4169E1', alpha=0.85, edgecolor='black', linewidth=0.5)
axes[1].bar(x - 0.5*width, dg_vals, width, label='DG', color='#32CD32', alpha=0.85, edgecolor='black', linewidth=0.5)
axes[1].bar(x + 0.5*width, ca_vals, width, label='CA Clash', color='#DC143C', alpha=0.85, edgecolor='black', linewidth=0.5)
axes[1].bar(x + 1.5*width, sc_vals, width, label='SC Clash', color='#FF8C00', alpha=0.85, edgecolor='black', linewidth=0.5)
axes[1].plot(x, total_vals, 'ko-', linewidth=2.5, markersize=10, label='Total', zorder=10, markerfacecolor='white', markeredgewidth=2)

axes[1].set_xlabel('Time t', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Loss Value', fontsize=11, fontweight='bold')
axes[1].set_title('Loss Evolution Over Time', fontsize=12, fontweight='bold', pad=18)
axes[1].set_xticks(x)
axes[1].set_xticklabels([f'{tp:.2f}' for tp in time_points], fontsize=10)
# Move legend to avoid overlap
axes[1].legend(fontsize=8, loc='upper left', framealpha=0.95, ncol=3, columnspacing=0.8)
axes[1].grid(True, alpha=0.3, axis='y', linestyle='--')
axes[1].set_axisbelow(True)

# Weight functions
axes[2].plot(t, np.ones_like(t), '#4169E1', linewidth=2.5, label='Velocity Weight (constant)', alpha=0.9)
axes[2].plot(t, t * max_dg_weight, '#32CD32', linewidth=2.5, label=f'DG Weight (×{max_dg_weight})', alpha=0.9)
axes[2].plot(t, t * max_clash_weight, '#DC143C', linewidth=2.5, label=f'Clash Weight (×{max_clash_weight})', alpha=0.9)
axes[2].fill_between(t, 0, np.ones_like(t), alpha=0.15, color='#4169E1')
axes[2].fill_between(t, 0, t * max_dg_weight, alpha=0.15, color='#32CD32')
axes[2].fill_between(t, 0, t * max_clash_weight, alpha=0.15, color='#DC143C')
axes[2].set_xlabel('Time t', fontsize=11, fontweight='bold')
axes[2].set_ylabel('Weight Value', fontsize=11, fontweight='bold')
axes[2].set_title('Time-Dependent Weighting', fontsize=12, fontweight='bold', pad=18)
axes[2].legend(fontsize=8, loc='upper left', framealpha=0.95)
axes[2].grid(True, alpha=0.3, linestyle='--')
axes[2].set_xlim(-0.02, 1.02)
axes[2].set_ylim(-0.01, 0.12)

plt.tight_layout(pad=3.0)
plt.savefig('loss_breakdown.png', dpi=150, bbox_inches='tight', facecolor='white')
print("✓ Saved loss_breakdown.png")

# ==================== Figure 3: Formula Visualization ====================
fig3 = plt.figure(figsize=(15, 10))
ax = fig3.add_subplot(111)
ax.axis('off')
fig3.patch.set_facecolor('white')

formulas = [
    ('Velocity Loss', 
     r'$L_{vel} = MSE(v_{pred}, v_{true})$',
     r'$v_{true} = x_1 - x_0$ (constant for linear path)',
     'No time weighting'),
    
    ('Distance Geometry Loss',
     r'$L_{DG} = \frac{1}{B} \sum [ReLU(d_{lower} - d_{ij})^2 + ReLU(d_{ij} - d_{upper})^2]$',
     r'Weighted by: $w(t) = t \times 0.1$ (default: $\lambda_{DG} = 0.1$)',
     'Time-weighted constraint: t=1 → weight=0.1'),
    
    ('CA Clash Loss',
     r'$L_{CA} = \frac{1}{B} \sum ReLU(\tau_{CA} + m - d_{ij})^2 \times w(t)$',
     r'$\tau_{CA} = 3.0$ Å, $m = 1.0$ Å, $w(t) = t \times 0.1$ (default)',
     'Time-weighted constraint: t=1 → weight=0.1'),
    
    ('SC Clash Loss',
     r'$L_{SC} = \frac{1}{B} \sum ReLU(\tau_{SC} + m - d_{ij})^2 \times w(t)$',
     r'$\tau_{SC} = 2.5$ Å, $m = 1.0$ Å, $w(t) = t \times 0.1$ (default)',
     'Time-weighted constraint: t=1 → weight=0.1'),
    
    ('Total Loss',
     r'$L_{total} = L_{vel} + L_{DG} + L_{CA} + L_{SC}$',
     r'Scaled for gradient accumulation: $L_{scaled} = L_{total} / N_{acc}$',
     'Sum of all components')
]

# Increase spacing between formulas - use manual positioning to avoid overlap
y_positions = [0.92, 0.72, 0.52, 0.32, 0.12]  # Manual positions for each formula
colors_list = ['#4169E1', '#32CD32', '#DC143C', '#FF8C00', '#000000']

# Calculate proper spacing
title_height = 0.05
formula1_height = 0.04
formula2_height = 0.035
note_height = 0.03
spacing_between = 0.015

for i, (title, formula1, formula2, note) in enumerate(formulas):
    y_top = y_positions[i]
    color = colors_list[i]
    
    # Title with background - positioned at top
    ax.text(0.08, y_top, title, fontsize=15, fontweight='bold', color=color,
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.7', facecolor='white', 
                     edgecolor=color, linewidth=2.5, alpha=0.95))
    
    # Formula 1 - positioned below title
    y_formula1 = y_top - title_height - spacing_between
    ax.text(0.08, y_formula1, formula1, fontsize=13,
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='white', 
                     edgecolor=color, linewidth=2, alpha=0.8))
    
    # Formula 2 - positioned below formula1
    y_formula2 = y_formula1 - formula1_height - spacing_between
    if formula2:
        ax.text(0.08, y_formula2, formula2, fontsize=11, style='italic',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.6', facecolor='#F8F8F8', 
                         edgecolor='gray', linewidth=1.2, alpha=0.7))
    
    # Note - positioned below formula2
    y_note = y_formula2 - formula2_height - spacing_between
    ax.text(0.08, y_note, f'Note: {note}', fontsize=10, alpha=0.85,
            transform=ax.transAxes, verticalalignment='top',
            style='italic')

ax.set_title('Loss Implementation Formulas (train_multi.py)', 
             fontsize=17, fontweight='bold', pad=30)

plt.tight_layout(pad=3.0)
plt.savefig('loss_formulas.png', dpi=150, bbox_inches='tight', facecolor='white')
print("✓ Saved loss_formulas.png")

print("\n✓ All visualizations saved!")
print("  - loss_visualization.png: Main loss plots")
print("  - loss_breakdown.png: Component breakdown")
print("  - loss_formulas.png: Mathematical formulas")
