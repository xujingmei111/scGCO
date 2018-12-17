library('AnnotationHub')
library('clusterProfiler')
library('org.Hs.eg.db')
library('ggplot2')

## ---BC---
## de=read.csv('scGCO_BC_min3.csv', sep=',',stringsAsFactors = F)

# --MOB---
de=read.csv('scGCO_MOB_min6.csv', sep=',',stringsAsFactors = F)

de_id<-bitr(de$gene,fromType = 'SYMBOL',toType = 'ENTREZID',
             OrgDb =org.Hs.eg.db )
de_GO_BP=enrichGO(de_id$ENTREZID,OrgDb = org.Hs.eg.db,keyType = 'ENTREZID',
                pvalueCutoff = 0.01,qvalueCutoff = 0.01,pAdjustMethod = 'fdr',
                readable = T,ont = 'BP')  
de_GO_BP_s1=simplify(de_GO_BP,cutoff=0.4, by='p.adjust',select_fun=min)    ## cutoff larger,more strict
de_GO_BP_s_m=as.data.frame(de_GO_BP_s1)
de_GO_BP_s_m=de_GO_BP_s_m[order(de_GO_BP_s_m$qvalue),]


de_GO_CC=enrichGO(de_id$ENTREZID,OrgDb = org.Hs.eg.db,keyType = 'ENTREZID',
                  pvalueCutoff = 0.01,qvalueCutoff = 0.01,pAdjustMethod = 'fdr',
                  readable = T,ont = 'CC') 
de_GO_CC_s=simplify(de_GO_CC,cutoff=0.4, by='p.adjust',select_fun=min)
de_GO_CC_s_m=as.data.frame(de_GO_CC_s)
de_GO_CC_s_m=de_GO_BP_s_m[order(de_GO_CC_s_m$qvalue),]


de_GO_MF=enrichGO(de_id$ENTREZID,OrgDb = org.Hs.eg.db,keyType = 'ENTREZID',
                  pvalueCutoff = 0.01,qvalueCutoff = 0.01,pAdjustMethod = 'fdr',
                  readable = T,ont = 'MF') 
de_GO_MF_s=simplify(de_GO_MF,cutoff=0.4, by='p.adjust',select_fun=min)
de_GO_MF_s_m=as.data.frame(de_GO_MF_s)
de_GO_MF_s_m=de_GO_BP_s_m[order(de_GO_MF_s_m$qvalue),]


df_GO=data.frame(ID=c(de_GO_BP_s_m$ID,de_GO_CC_s_m$ID,de_GO_MF_s_m$ID),
                 Description=c(de_GO_BP_s_m$Description,de_GO_CC_s_m$Description,
                               de_GO_MF_s_m$Description),
                 GeneNumber=c(de_GO_BP_s_m$Count,de_GO_CC_s_m$Count,de_GO_MF_s_m$Count),
                 type=factor(c(rep("Biological Process", dim(de_GO_BP_s_m)[1]), 
                               rep("Cellular Component", dim(de_GO_CC_s_m)[1]), 
                               rep("Molecular Function", dim(de_GO_MF_s_m)[1])), 
                             levels=c("Biological Process", "Cellular Component",
                                      "Molecular Function")))
                                     

df_GO$order=factor(1:nrow(df_GO))

# shorten the names of GO terms
shorten_names <- function(x, n_word=4){ 
  if (length(strsplit(x, " ")[[1]]) > n_word ) { 
    #if (nchar(x) > 40) x <- substr(x, 1, 40) 
    x <- paste(paste(strsplit(x, " ")[[1]][1:min(length(strsplit(x," ")[[1]]), n_word)], 
                     collapse=" "), sep="") 
    return(x) 
}
  else 
  { 
      return(x) 
    }
}

labels=(sapply(
  levels(df_GO$Description)[as.numeric(df_GO$Description)],
  shorten_names))

names(labels) = 1:nrow(df_GO)

CPCOLS <- c("#8DA1CB", "#FD8D62", "#66C3A5")

p <- ggplot(data=df_GO, aes(x=order, y=GeneNumber, fill=type))+
  geom_bar(stat="identity",position = "dodge",width = 0.6)+
  scale_fill_manual('Sub-ontology',values = CPCOLS) +       # legend title ,color 
  theme_bw() + scale_x_discrete(labels=labels)  + 
  theme(axis.text=element_text(color="black")) +
  labs(title = "GO Terms of MOB_data with spatialDE ",x='')+
  ylab('Number of genes in the term')+
  theme(axis.text.x = element_text(angle = 50,hjust = 1,size=10),
        axis.title.y = element_text(size = 9),
          legend.position = 'left',
        legend.key.size = unit(2,'mm'),
        legend.title = element_text(size=9,face = 'bold'),
        legend.text = element_text(size=8))+
    theme(panel.grid =element_blank())+ 
  theme(panel.border = element_blank())+
  theme(axis.line.x.bottom  = element_line(size = 0.5,colour = 'gray50'),
       axis.line.y.left =element_line(size = 0.5,colour = 'gray50') ,
       plot.title = element_text(hjust = 0.5))

p



