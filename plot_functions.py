import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.ticker as ticker

def plot_method(df, method, feat1, feat2, feat3=None):
    
    if feat3==None:
        plt.figure()
        sns.scatterplot(x=df[feat1], y=df[feat2], hue=df[method])
        plt.title(method)
    else:
        plt.figure()
        ax = plt.axes(projection="3d")
        
        if df[method].dtype == 'O':
            ax.scatter(df[feat1], df[feat2], df[feat3], c=df[method].apply(ord))
        else:
            ax.scatter(df[feat1], df[feat2], df[feat3], c=df[method])
        
        plt.title(method)
        ax.set_xlabel(feat1)
        ax.set_ylabel(feat2)
        ax.set_zlabel(feat3)
        
def plot_count(df,feat,save=False, directory=None):
    
    plt.figure(figsize=(12,8))
    ax = sns.countplot(x=feat, data=df, order=sorted(df[feat].unique()))
    plt.xlabel(feat)
    
    ncount = len(df)
    # Make twin axis
    ax2=ax.twinx()
    
    # Switch so count axis is on right, frequency on left
    ax2.yaxis.tick_left()
    ax.yaxis.tick_right()
    
    # Switch labels as well
    ax.yaxis.set_label_position('right')
    ax2.yaxis.set_label_position('left')
    
    ax2.set_ylabel('Frequency [%]')
    
    for p in ax.patches:
        x=p.get_bbox().get_points()[:,0]
        y=p.get_bbox().get_points()[1,1]
        ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 
                ha='center', va='bottom') # set the alignment of the text
    
    # Use a LinearLocator to ensure the correct number of ticks
    ax.yaxis.set_major_locator(ticker.LinearLocator(11))
    
    # Fix the frequency range to 0-100
    ax2.set_ylim(0,100)
    ax.set_ylim(0,ncount)
    
    # And use a MultipleLocator to ensure a tick spacing of 10
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))
    
    # Need to turn the grid on ax2 off, otherwise the gridlines end up on top of the bars
    ax2.grid(None)
    
    plt.show()
    
    if save==True:
        plt.savefig(directory)
 
       
def plot_bar(df,feat_x,feat_y,save=False, directory=None):
    
    plt.figure(figsize=(12,8))    
    ax = sns.barplot(x=feat_x,y=feat_y,data=df,ci=None,
                     estimator=sum,order=sorted(df[feat_x].unique()))
    plt.xlabel(feat_x)
    
    for p in ax.patches:
        x=p.get_bbox().get_points()[:,0]
        y=p.get_bbox().get_points()[1,1]
        ax.annotate('{:,.2f}'.format(y), (x.mean(), y), 
                ha='center', va='bottom') # set the alignment of the text
        
    plt.show()
    
    if save==True:
        plt.savefig(directory)        