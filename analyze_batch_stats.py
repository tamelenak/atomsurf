import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import argparse

def load_and_clean_data(csv_path):
    """Load and clean the batch stats data"""
    df = pd.read_csv(csv_path)
    
    # Convert protein_names to lists
    df['protein_list'] = df['protein_names'].apply(lambda x: x.split(';') if pd.notna(x) and x != '' else [])
    df['num_proteins'] = df['protein_list'].apply(len)
    
    # Remove rows with missing data
    df = df.dropna(subset=['loss'])
    
    return df

def analyze_protein_specific_loss(df):
    """Analyze loss statistics for each individual protein"""
    protein_losses = defaultdict(list)
    
    for _, row in df.iterrows():
        loss = row['loss']
        for protein in row['protein_list']:
            protein_losses[protein].append(loss)
    
    # Calculate statistics for each protein
    protein_stats = {}
    for protein, losses in protein_losses.items():
        if len(losses) >= 5:  # Only consider proteins with enough samples
            protein_stats[protein] = {
                'mean_loss': np.mean(losses),
                'std_loss': np.std(losses),
                'min_loss': np.min(losses),
                'max_loss': np.max(losses),
                'count': len(losses),
                'cv': np.std(losses) / np.mean(losses) if np.mean(losses) > 0 else 0  # Coefficient of variation
            }
    
    return protein_stats

def analyze_batch_composition_effects(df):
    """Analyze how batch composition affects loss"""
    results = {}
    
    # 1. Batch size vs loss
    batch_size_corr = df[['batch_size', 'loss']].corr().iloc[0, 1]
    results['batch_size_correlation'] = batch_size_corr
    
    # 2. Vertex count vs loss
    vertex_corr = df[['vertex_count', 'loss']].corr().iloc[0, 1]
    results['vertex_count_correlation'] = vertex_corr
    
    # 3. Node count vs loss
    node_corr = df[['node_count', 'loss']].corr().iloc[0, 1]
    results['node_count_correlation'] = node_corr
    
    # 4. Loss stability over epochs
    epoch_loss_std = df.groupby('epoch')['loss'].std().mean()
    results['average_epoch_loss_std'] = epoch_loss_std
    
    return results

def find_problematic_proteins(protein_stats, threshold_percentile=90):
    """Find proteins that consistently cause high loss"""
    mean_losses = [stats['mean_loss'] for stats in protein_stats.values()]
    threshold = np.percentile(mean_losses, threshold_percentile)
    
    problematic = {protein: stats for protein, stats in protein_stats.items() 
                   if stats['mean_loss'] > threshold}
    
    return problematic

def analyze_batch_diversity_effect(df):
    """Analyze if batch diversity (unique proteins) affects loss stability"""
    diversity_analysis = []
    
    for _, row in df.iterrows():
        proteins = row['protein_list']
        unique_proteins = len(set(proteins))
        total_proteins = len(proteins)
        diversity_ratio = unique_proteins / total_proteins if total_proteins > 0 else 0
        
        diversity_analysis.append({
            'batch_idx': row['batch_idx'],
            'epoch': row['epoch'],
            'loss': row['loss'],
            'unique_proteins': unique_proteins,
            'total_proteins': total_proteins,
            'diversity_ratio': diversity_ratio
        })
    
    diversity_df = pd.DataFrame(diversity_analysis)
    diversity_corr = diversity_df[['diversity_ratio', 'loss']].corr().iloc[0, 1]
    
    return diversity_df, diversity_corr

def create_visualizations(df, protein_stats, save_prefix="batch_analysis"):
    """Create visualizations for the analysis"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Loss over time
    axes[0, 0].plot(df['batch_idx'], df['loss'], alpha=0.6)
    axes[0, 0].set_title('Loss Over Batches')
    axes[0, 0].set_xlabel('Batch Index')
    axes[0, 0].set_ylabel('Loss')
    
    # 2. Loss vs batch size
    axes[0, 1].scatter(df['batch_size'], df['loss'], alpha=0.6)
    axes[0, 1].set_title('Loss vs Batch Size')
    axes[0, 1].set_xlabel('Batch Size')
    axes[0, 1].set_ylabel('Loss')
    
    # 3. Loss vs vertex count
    axes[0, 2].scatter(df['vertex_count'], df['loss'], alpha=0.6)
    axes[0, 2].set_title('Loss vs Vertex Count')
    axes[0, 2].set_xlabel('Vertex Count')
    axes[0, 2].set_ylabel('Loss')
    
    # 4. Distribution of loss
    axes[1, 0].hist(df['loss'], bins=50, alpha=0.7)
    axes[1, 0].set_title('Loss Distribution')
    axes[1, 0].set_xlabel('Loss')
    axes[1, 0].set_ylabel('Frequency')
    
    # 5. Top proteins by mean loss
    if protein_stats:
        top_proteins = sorted(protein_stats.items(), key=lambda x: x[1]['mean_loss'], reverse=True)[:10]
        protein_names = [p[0][:10] + '...' if len(p[0]) > 10 else p[0] for p in top_proteins]  # Truncate names
        mean_losses = [p[1]['mean_loss'] for p in top_proteins]
        
        axes[1, 1].barh(protein_names, mean_losses)
        axes[1, 1].set_title('Top 10 Proteins by Mean Loss')
        axes[1, 1].set_xlabel('Mean Loss')
    
    # 6. Loss standard deviation by epoch
    epoch_std = df.groupby('epoch')['loss'].std()
    axes[1, 2].plot(epoch_std.index, epoch_std.values)
    axes[1, 2].set_title('Loss Std Dev by Epoch')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Loss Std Dev')
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_overview.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_report(df, protein_stats, batch_effects, problematic_proteins, diversity_corr):
    """Generate a text report with findings"""
    report = []
    report.append("=== BATCH COMPOSITION ANALYSIS REPORT ===\n")
    
    report.append(f"Total batches analyzed: {len(df)}")
    report.append(f"Total unique proteins: {len(protein_stats)}")
    report.append(f"Average loss: {df['loss'].mean():.4f} ± {df['loss'].std():.4f}")
    report.append(f"Loss range: {df['loss'].min():.4f} - {df['loss'].max():.4f}\n")
    
    report.append("=== CORRELATIONS ===")
    report.append(f"Batch size vs Loss: {batch_effects['batch_size_correlation']:.4f}")
    report.append(f"Vertex count vs Loss: {batch_effects['vertex_count_correlation']:.4f}")
    report.append(f"Node count vs Loss: {batch_effects['node_count_correlation']:.4f}")
    report.append(f"Batch diversity vs Loss: {diversity_corr:.4f}\n")
    
    report.append("=== PROBLEMATIC PROTEINS (Top 90th percentile by mean loss) ===")
    for protein, stats in sorted(problematic_proteins.items(), key=lambda x: x[1]['mean_loss'], reverse=True)[:10]:
        report.append(f"{protein}: mean={stats['mean_loss']:.4f}, std={stats['std_loss']:.4f}, count={stats['count']}")
    
    # Identify most variable proteins
    most_variable = sorted(protein_stats.items(), key=lambda x: x[1]['cv'], reverse=True)[:5]
    report.append("\n=== MOST VARIABLE PROTEINS (by coefficient of variation) ===")
    for protein, stats in most_variable:
        report.append(f"{protein}: CV={stats['cv']:.4f}, mean={stats['mean_loss']:.4f}, std={stats['std_loss']:.4f}")
    
    return "\n".join(report)

def main():
    parser = argparse.ArgumentParser(description='Analyze batch composition effects on training loss')
    parser.add_argument('csv_path', default='batch_stats.csv', help='Path to batch_stats.csv file')
    parser.add_argument('--save_prefix', default='analysis/batch_analysis', help='Prefix for saved files')
    
    args = parser.parse_args()
    
    print("Loading and cleaning data...")
    df = load_and_clean_data(args.csv_path)
    
    print("Analyzing protein-specific loss patterns...")
    protein_stats = analyze_protein_specific_loss(df)
    
    print("Analyzing batch composition effects...")
    batch_effects = analyze_batch_composition_effects(df)
    
    print("Finding problematic proteins...")
    problematic_proteins = find_problematic_proteins(protein_stats)
    
    print("Analyzing batch diversity effects...")
    diversity_df, diversity_corr = analyze_batch_diversity_effect(df)
    
    print("Creating visualizations...")
    create_visualizations(df, protein_stats, args.save_prefix)
    
    print("Generating report...")
    report = generate_report(df, protein_stats, batch_effects, problematic_proteins, diversity_corr)
    
    # Save report
    with open(f'{args.save_prefix}_report.txt', 'w') as f:
        f.write(report)
    
    print("\n" + report)
    
    # Save detailed protein stats
    protein_df = pd.DataFrame.from_dict(protein_stats, orient='index')
    protein_df.to_csv(f'{args.save_prefix}_protein_stats.csv')
    
    # Save diversity analysis
    diversity_df.to_csv(f'{args.save_prefix}_diversity_analysis.csv', index=False)
    
    print(f"\nFiles saved:")
    print(f"- {args.save_prefix}_report.txt")
    print(f"- {args.save_prefix}_protein_stats.csv")
    print(f"- {args.save_prefix}_diversity_analysis.csv")
    print(f"- {args.save_prefix}_overview.png")

if __name__ == "__main__":
    main() 