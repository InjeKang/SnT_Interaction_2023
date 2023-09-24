setwd("D:/Analysis/2023_LSSB")

# load required packages
source("r/functions_variables.R")
load_package(packages)

# load Python environment
Sys.setenv(PYTHONWARNINGS = "ignore")
use_condaenv(condaenv = "knowledgeNetwork", required = TRUE)

# load the data
pd = import("pandas")
data <- pd$read_pickle("data/03.stm_data.pkl")

# Covert year into date object and integer
data$year_int = data$year - 1999

data = data %>%
  mutate(year = as.numeric(year)) %>%
  filter(year >= 2000 & year <= 2020)

# Sort by date
data = data %>% 
  arrange(year)

# Factorize
data$type = as.factor(data$country)


## 1. Annual scientific and technological activities.
china <- data %>%
  dplyr::filter(type == "CN") %>%
  dplyr::group_by(year) %>%
  dplyr::summarise(count = n())

japan <- data %>%
  dplyr::filter(type == "JP") %>%
  dplyr::group_by(year) %>%
  dplyr::summarize(count = n())

patent_activities <- left_join(china, japan, by = "year") %>%
  mutate(china = count.x, japan = count.y, ratio = count.x / count.y) %>%
  select(year, china, japan, ratio)

china_activities_plot = ggplot(patent_activities, aes(x = year, y = china)) +
  geom_line() +
  scale_x_continuous(breaks = seq(min(patent_activities$year), max(patent_activities$year), by = 2)) +
  labs(title = "Annual patent activities in China", x = "Year", y = "Patent applications")

japan_activities_plot = ggplot(patent_activities, aes(x = year, y = japan)) +
  geom_line() +
  scale_x_continuous(breaks = seq(min(patent_activities$year), max(patent_activities$year), by = 2)) +
  labs(title = "Annual patent activities in Japan", x = "Year", y = "Patent applications")

write.csv(patent_activities, file = "stm/1.patent_activities.csv", row.names = FALSE)
# Save the plot as a PNG file
ggsave("stm/01.patent_activities_plot_in_China.png", plot = china_activities_plot, width = 7, height = 5)
ggsave("stm/01.patent_activities_plot_in_Japan.png", plot = japan_activities_plot, width = 7, height = 5)

# Tokenize
data = data.frame(data[c("type", "id", "year_int", "corpus_cleansed")])

myprocess = textProcessor(data$corpus_cleansed, metadata = data, stem=FALSE)
out = prepDocuments(myprocess$documents, myprocess$vocab, myprocess$meta, lower.thresh = as.integer(length(data$text) * 0.001))

# Find optinal number of K
search_k = T
K_values <- 5:30
if (!search_k) {
  model_searchK <- searchK(out$documents, out$vocab, K = K_values,      
                           prevalence = ~type * s(year_int),
                           data = out$meta, seed = 2022)
  
  model_name <- paste0("stm/02.model_searchK_", K_values[1], "-", K_values[length(K_values)])
  saveRDS(model_searchK, paste0(model_name, ".rds"))
  plot(model_searchK)
} else {
  model_name <- paste0("stm/02.model_searchK_", K_values[1], "-", K_values[length(K_values)])
  model_searchK <- readRDS(paste0(model_name, ".rds"))
}

png("stm/03.model_searchK1.png", width = 750, height = 500)
plot(model_searchK)
dev.off()

png("stm/03.model_searchK2.png", width = 1000, height = 1000)
model_candidates = data.frame(sapply(model_searchK$result[,2:3], function(x) as.numeric(x)))
rownames(model_candidates) = paste0("K=", unlist(model_searchK$result[,1]))
ggplot(data=model_candidates, mapping=aes(x=semcoh, y=exclus)) +
  geom_text_repel(label=rownames(model_candidates), size=4) +
  labs(title="Semantic Coherence-Exclusivity Plot by Number of Topics(K)", 
       x="Semantic Coherence",
       y="Exclusivity") +
  theme_classic() 
dev.off()


# Create subfolder
num_topic <- 10 # 9, 10, 11
subfolder_name <- paste0("stm/topics_", num_topic)
dir.create(subfolder_name, showWarnings = FALSE)
dir.create(paste0(subfolder_name, "/doc_by_topics"), showWarnings = FALSE)

# Train K topics
model = T
if (model == F) {
  stm_model = stm(out$documents, out$vocab, K=num_topic,
                  prevalence= ~type*s(year_int),
                  data=out$meta, init.type="Spectral",seed=2022,
                  verbose = F)
  model_name = paste0(subfolder_name, "/topics_", num_topic ,'.rds')
  saveRDS(stm_model, model_name)
} else {
  model_name = paste0(subfolder_name, "/topics_", num_topic ,'.rds')
  stm_model = readRDS(model_name)
}

# Summary
topic_keywords = labelTopics(stm_model, n=5)

# Create dataframe
topic_keywords_df = do.call(cbind.data.frame, topic_keywords)
topic_keywords_df = topic_keywords_df %>%
  unite("PROB", paste0("prob.",1:5), sep=" ") %>%
  unite("FREX", paste0("frex.",1:5), sep=" ") %>%
  unite("LIFT", paste0("lift.",1:5), sep=" ") %>%
  unite("SCORE", paste0("score.",1:5), sep=" ")
write.csv(topic_keywords_df, paste0(subfolder_name, "/2.keywords_by_topics", num_topic, ".csv"), fileEncoding = "UTF-8")


## 2. Topic proportions
# make.dt(stm_model)
png(paste0(subfolder_name, "/2.top_topics_", num_topic ,".png"), width = 750, height = 500)
par(mfrow=c(1,1))
# Topic proportions plot
plot(stm_model, type='summary', labeltype = 'frex', n=5)
# Custom labels bar plot
# barplot(rep(1, length(topic_labels)), names.arg = topic_labels, horiz = TRUE, xlab = "Proportion", col = "lightblue")
# text(1:num_topic, par("usr")[3] - 0.03, labels = topic_labels, srt = 45, adj = c(1, 1), xpd = TRUE)
dev.off()

## 3. Network of topic correlation
# stm_model_corr <- topicCorr(stm_model, cutoff=0.05)
# png(paste0(subfolder_name, "/3.topic_corr_", num_topic ,".png"), width = 1000, height = 1000, res=300)
# par(mar = c(.5, .5, .5, .5))
# plot(stm_model_corr, vertex.size = 8, vertex.label.cex = 0.5, cex = 0.8) 
# dev.off()

# Estimate effect
stm_effect_model = estimateEffect(1:num_topic ~type*s(year_int), stm_model, meta = out$meta, uncertainty = "Global")
# summary(stm_effect_model, topics = 1)

# Topic labeling
# topic_labels = paste0("T", 1:num_topic)
topic_labels = paste0("T", 1:num_topic, "-", c("secondary battery",
                                               "lithium manufacturing process",
                                               "secondary battery parts manufacturing process",
                                               "electolyte",
                                               "cathode",
                                               "composite solid electrolyte (CSE)",
                                               "electric conducting materials",
                                               "lithium metal battery",
                                               "all solid-state battery (ASSB)",
                                               "sulfide electrolyte"))

# Document by topics for network analysis between topics
doc_topic = make.dt(stm_model, meta = NULL)
colnames(doc_topic)[2:11] <- topic_labels
# doc_topic[, top_topics := names(.SD)[order(-.SD)][1:5], .SDcols = 2:(num_topic+1)]
doc_topic$id = out$meta$id

write.xlsx(doc_topic, paste0(subfolder_name, "/3.doc_by_topics", num_topic, ".xlsx"))


## 4. Difference in topics
val1 <- unique(data$type)[1]
val2 <- unique(data$type)[2]

# Set the plot size and margins
png(paste0(subfolder_name, "/4.difference_in_topics", num_topic, ".png"), width = 750, height = 500)
par(mfrow = c(1, 1))
par(mar = c(5, 4, 4, 2) + 0.1) 

effect_plot <- plot.estimateEffect(stm_effect_model, covariate = "type",
                                   topics = c(1:num_topic), method = "difference",
                                   model = stm_model, 
                                   main = '',
                                   cov.value1 = val2, cov.value2 = val1,
                                   xlab = paste(val1, "vs.", val2),
                                   xlim = c(-0.5, 0.5),
                                   labeltype = "custom", n = 5,
                                   width = 100, verbose.labels = FALSE,
                                   custom.labels = topic_labels)

dev.off()

# # Get topic prevalence for each combination
# effects <- data %>%
#   mutate(topic_prevalence = map2_dbl(year_int, type, function(year, country) {
#     subset <- get_effects(
#       estimates = stm_effect_model,
#       variable = 'year_int',
#       type = 'continuous',
#       moderator = 'type',
#       modval = country
#     )
#     result <- subset$effects[subset$effects$year_int == year, "estimate"]
#     if (length(result) == 0) result <- 0 # Set prevalence to 0 for missing data
#     result
#   }))


# Topic prevalence
effects = get_effects(estimates = stm_effect_model,
                      variable = 'year_int',
                      type = 'continuous',
                      moderator = 'type',
                      modval = 'CN') %>%
  bind_rows(
    get_effects(estimates = stm_effect_model,
                variable = 'year_int',
                type = 'continuous',
                moderator = 'type',
                modval = 'JP'))

# # normalize the values of proportion
# # Min-max scaling
# min_max_normalize <- function(x) {
#   (x - min(x)) / (max(x) - min(x))
# }
# 
# effects$proportion_norm <- min_max_normalize(effects$proportion)


# Sigmoid normalization
sigmoid_normalize <- function(x) {
  1 / (1 + exp(-x))
}

effects$proportion_norm <- sigmoid_normalize(effects$proportion)

# # Z-score normalization
# z_score_normalize <- function(x) {
#   (x - mean(x)) / sd(x)
# }
# 
# effects$proportion_norm <- z_score_normalize(effects$proportion)

# knots = c(attr(stm_effect_model$modelframe$`s(year_int)`, 'Boundary.knots')[1], as.vector(attr(stm_effect_model$modelframe$`s(year_int)`, 'knots')), attr(stm_effect_model$modelframe$`s(year_int)`, 'Boundary.knots')[2])
years_names <- seq(2000, 2020)
years_ints <- seq(1:21)
years_map = years_names[years_ints]
plot_list = list()
for (t in c(1:num_topic)) {
  effects_topic = filter(effects, topic == t)
  effects_topic = mutate(effects_topic, moderator = as.factor(effects_topic$moderator))
  p = ggplot(effects_topic, aes(x = value, y = proportion_norm, color = moderator,
                                group = moderator, fill = moderator)) +
    geom_line() +
    # geom_hline(yintercept = 0, size=0.75) +
    # geom_vline(xintercept = knots, size=0.5, color="darkgrey", linetype="dashed") +
    # geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.2)  +
    theme(legend.position = "none") +
    # labs(title=paste0(topic_labels[t]), x = 'Year', y = 'Topic Proportion', color = 'type', group = 'type', fill = 'type') +
    labs(t, x = 'Year', y = 'Expected Topic Proportion', color = 'type', group = 'type', fill = 'type') +
    scale_x_continuous(name="Year", 
                       breaks = seq(min(effects_topic$value), max(effects_topic$value), length.out=length(years_names)),
                       labels = years_map) +
    theme_classic()+
    theme(axis.text.x = element_text(angle=90, size=rel(0.9), hjust=1, vjust=0.5),
          legend.box.margin = margin(2.5,unit="pt"))
  plot_list[[t]] = p
}

ml = marrangeGrob(plot_list, nrow=1, ncol=1)
pdf.options(family = "Korea1deb")
ggsave(paste0(subfolder_name, "/5.stm_topic_prevalence", num_topic, ".pdf"), plot=ml, width = 12, height = 8, units="in")

## 5. Topic prevalence by periods
# Create a dataframe with the topic proportions for each document
doc_topics <- as.data.frame(stm_model$theta)

# Add the year variable to the dataframe
doc_topics$year <- out$meta$year_int

# Divide the period into four time periods
doc_topics$period <- cut(doc_topics$year, 
                         breaks = c(1, 6, 11, 16, 21), 
                         labels = c("2000-2005", "2006-2010", "2011-2015", "2016-2020"))

# Add the type variable to the dataframe
doc_topics$type <- out$meta$type

# Select the topics of interest
topics_of_interest <- 1:num_topic

# Get the labels of the selected topics
topic_labels_selected <- topic_labels[topics_of_interest]

# Create a dataframe of the top topics by period and type
topic_means <- aggregate(doc_topics[, topics_of_interest], 
                         by = list(period = doc_topics$period, type = doc_topics$type), 
                         FUN = mean)

topic_means$top_topics <- apply(topic_means[, paste0("V", topics_of_interest)], 1, function(x) {
  top_topics <- order(-x)
  paste0(topic_labels_selected[top_topics], " (", format(round(x[top_topics] * 100), trim = TRUE), "%)", collapse = ", ")
})

write.csv(topic_means[,c("period", "type", "top_topics")], paste0(subfolder_name, "/6.top_topics_by_periods", num_topic, ".csv"), row.names = FALSE)
