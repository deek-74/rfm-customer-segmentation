install.packages(c("readxl", "dplyr", "tidyr", "lubridate", "ggplot2",
                   "scales", "cluster", "factoextra", "ggrepel", "knitr",
                   "corrplot", "NbClust", "gridExtra"))
library(readxl)
library(dplyr)
library(tidyr)
library(lubridate)
library(ggplot2)
library(scales)
library(cluster)
library(factoextra)
library(ggrepel)
library(knitr)
library(corrplot)
library(gridExtra)
library(purrr)

df_raw<-read_excel("D:\\INTERNSHIPS\\Projects\\RFM\\Online Retail.xlsx")
glimpse(df_raw)
colSums(is.na(df_raw))
df<- df_raw
df<-unique(df)

df<-df[!is.na(df$CustomerID),]
df<-df[!is.na(df$Description),]
df <- df[substring(df$InvoiceNo, 1, 1) != "C", ]
df<- df[df$Quantity>0,]
df<- df[df$UnitPrice>0,]
df$CustomerID <- as.character(df$CustomerID)
df$Revenue<- df$Quantity * df$UnitPrice


cat("Rows after cleaning:  ", nrow(df), "\n")
cat("Unique customers:     ", length(unique(df$CustomerID)), "\n")
cat("Unique invoices:      ", length(unique(df$InvoiceNo)), "\n")
cat("Date range (from):    ", format(min(df$InvoiceDate)), "\n")
cat("Date range (to):      ", format(max(df$InvoiceDate)), "\n")

df_transactions<- df

# Dataset 2: Customer-level summary


# Reference date = 1 day after the last transaction in the data
reference_date <- max(df$InvoiceDate) + 86400   # 86400 seconds = 1 day
customer_ids<- unique(df$CustomerID)
df_customers<- data.frame(
  CustomerID= character(length(customer_ids)),
  Recency= numeric(length(customer_ids)),
  Frequency=numeric(length(customer_ids)),
  Monetary=numeric(length(customer_ids)),
  AvgOrderValue=numeric(length(customer_ids)),
  Country=character(length(customer_ids)),
  StringAsFactors=FALSE
)

for (i in seq_along(customer_ids)){
  cust_id <- customer_ids[i]
  cust_rows <- df[df$CustomerID==cust_id,]
  last_purchase<- max(cust_rows$InvoiceDate)
  num_orders<- length(unique(cust_rows$InvoiceNo))
  total_revenue<- sum(cust_rows$Revenue)
  df_customers$CustomerID[i]    <- cust_id
  df_customers$Recency[i]       <- as.numeric(difftime(reference_date,
                                                       last_purchase,
                                                       units = "days"))
  df_customers$Frequency[i]     <- num_orders
  df_customers$Monetary[i]      <- total_revenue
  df_customers$AvgOrderValue[i] <- total_revenue / num_orders
  df_customers$Country[i]       <- cust_rows$Country[1]
}

head(df_customers)
nrow(df_customers)


# ── Core KPIs ──────────────────────────────────────────────────────────────────
total_revenue      <- sum(df$Revenue)
total_orders       <- n_distinct(df$InvoiceNo)
total_customers    <- n_distinct(df$CustomerID)
avg_order_value    <- total_revenue / total_orders
revenue_per_cust   <- total_revenue / total_customers

repeat_customers   <- df_customers %>% filter(Frequency > 1) %>% nrow()
repeat_rate        <- repeat_customers / total_customers * 100


cat("  BUSINESS KPI SUMMARY\n")

cat(sprintf("  Total Revenue:        £%s\n",   comma(round(total_revenue, 0))))
cat(sprintf("  Total Orders:         %s\n",    comma(total_orders)))
cat(sprintf("  Total Customers:      %s\n",    comma(total_customers)))
cat(sprintf("  Avg Order Value:      £%.2f\n", avg_order_value))
cat(sprintf("  Revenue per Customer: £%.2f\n", revenue_per_cust))
cat(sprintf("  Repeat Purchase Rate: %.1f%%\n", repeat_rate))

# ── Monthly Revenue ─────────────────────────────────────────────────────────────
monthly_revenue <- df %>%
  mutate(YearMonth = floor_date(InvoiceDate, "month")) %>%
  group_by(YearMonth) %>%
  summarise(Revenue = sum(Revenue), Orders = n_distinct(InvoiceNo))

# ── Top Countries ───────────────────────────────────────────────────────────────
top_countries <- df %>%
  group_by(Country) %>%
  summarise(Revenue = sum(Revenue)) %>%
  arrange(desc(Revenue)) %>%
  slice_head(n = 10)

# ── Top Products ────────────────────────────────────────────────────────────────
top_products <- df %>%
  group_by(Description) %>%
  summarise(Revenue = sum(Revenue)) %>%
  arrange(desc(Revenue)) %>%
  slice_head(n = 10)

# ── Purchase Frequency Distribution ────
freq_dist <- df_customers %>%
  count(Frequency) %>%
  arrange(Frequency)

print(monthly_revenue)
print(top_countries)
print(top_products)


theme_retail <- theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold", size = 14),
        plot.subtitle = element_text(color = "grey50"),
        panel.grid.minor = element_blank())

p1 <- ggplot(monthly_revenue, aes(x = YearMonth, y = Revenue)) +
  geom_line(color = "#2E86AB", linewidth = 1.2) +
  geom_point(color = "#2E86AB", size = 2.5) +
  scale_y_continuous(labels = dollar_format(prefix = "£")) +
  scale_x_datetime(date_labels = "%b %Y", date_breaks = "1 month") +
  labs(title = "Monthly Revenue Trend", subtitle = "Dec 2010 – Dec 2011",
       x = NULL, y = "Revenue") +
  theme_retail +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
print(p1)


p2 <- top_products %>%
  mutate(Description = stringr::str_wrap(Description, 30)) %>%
  ggplot(aes(x = reorder(Description, Revenue), y = Revenue)) +
  geom_col(fill = "#A23B72", alpha = 0.85) +
  coord_flip() +
  scale_y_continuous(labels = dollar_format(prefix = "£")) +
  labs(title = "Top 10 Products by Revenue", x = NULL, y = "Revenue") +
  theme_retail
print(p2)


p3 <- top_countries %>%
  ggplot(aes(x = reorder(Country, Revenue), y = Revenue)) +
  geom_col(fill = "#F18F01", alpha = 0.85) +
  coord_flip() +
  scale_y_continuous(labels = dollar_format(prefix = "£")) +
  labs(title = "Top 10 Countries by Revenue", x = NULL, y = "Revenue") +
  theme_retail
print(p3)

p4 <- df_customers %>%
  filter(Frequency <= 20) %>%
  ggplot(aes(x = Frequency)) +
  geom_histogram(binwidth = 1, fill = "#3BB273", color = "white") +
  labs(title = "Customer Purchase Frequency Distribution",
       subtitle = "Showing customers with ≤ 20 orders",
       x = "Number of Orders", y = "Number of Customers") +
  theme_retail
print(p4)


p5 <- df_customers %>%
  ggplot(aes(x = Monetary)) +
  geom_histogram(bins = 50, fill = "#E84855", color = "white", alpha = 0.8) +
  scale_x_log10(labels = dollar_format(prefix = "£")) +
  labs(title = "Customer Revenue Distribution (Log Scale)",
       x = "Total Revenue (log)", y = "Count") +
  theme_retail
print(p5)


print(p1); print(p2); print(p3); print(p4); print(p5)




df_rfm <- df_customers %>%
  mutate(
    R_Score = ntile(-Recency,  5),   # lower recency = better = higher score
    F_Score = ntile(Frequency,  5),
    M_Score = ntile(Monetary,   5),
    RFM_Score = R_Score * 100 + F_Score * 10 + M_Score
  )


assign_segment <- function(r, f, m) {
  case_when(
    r >= 4 & f >= 4 & m >= 4                        ~ "Champions",
    r >= 3 & f >= 3                                  ~ "Loyal Customers",
    r >= 4 & f <= 2                                  ~ "New Customers",
    r >= 3 & f >= 2 & m >= 3                         ~ "Potential Loyalists",
    m >= 4 & f <= 2                                  ~ "Big Spenders",
    r <= 2 & f >= 3 & m >= 3                         ~ "At-Risk Customers",
    r <= 2 & f >= 2                                  ~ "Hibernating Customers",
    TRUE                                              ~ "Lost Customers"
  )
}

df_rfm <- df_rfm %>%
  mutate(Segment = assign_segment(R_Score, F_Score, M_Score))

names(df_rfm)

#rfm table print
rfm_customer_table <- df_rfm %>%
  select(
    CustomerID,
    Recency,
    Frequency,
    Monetary,
    R_Score,
    F_Score,
    M_Score,
    RFM_Score,
    Segment
  )

write.csv(
  rfm_customer_table,
  "rfm_customer_table.csv",
  row.names = FALSE
)

segment_summary <- df_rfm %>%
  group_by(Segment) %>%
  summarise(
    Count        = n(),
    Pct          = round(n() / nrow(df_rfm) * 100, 1),
    Avg_Recency  = round(mean(Recency), 0),
    Avg_Frequency= round(mean(Frequency), 1),
    Avg_Monetary = round(mean(Monetary), 0),
    Total_Revenue= round(sum(Monetary), 0)
  ) %>%
  arrange(desc(Total_Revenue))

kable(segment_summary, format = "pipe", big.mark = ",")



seg_colors <- c(
  "Champions"           = "#2E86AB",
  "Loyal Customers"     = "#3BB273",
  "Potential Loyalists" = "#A7C957",
  "New Customers"       = "#F4A261",
  "Big Spenders"        = "#A23B72",
  "At-Risk Customers"   = "#E84855",
  "Hibernating Customers"= "#F18F01",
  "Lost Customers"      = "#6B6B6B"
)
p_seg1 <- df_rfm %>%
  count(Segment) %>%
  ggplot(aes(x = reorder(Segment, n), y = n, fill = Segment)) +
  geom_col(show.legend = FALSE) +
  coord_flip() +
  scale_fill_manual(values = seg_colors) +
  labs(title = "Customer Count by RFM Segment", x = NULL, y = "Customers") +
  theme_retail
print(p_seg1)

p_seg2 <- df_rfm %>%
  group_by(Segment) %>%
  summarise(Revenue = sum(Monetary)) %>%
  ggplot(aes(x = reorder(Segment, Revenue), y = Revenue, fill = Segment)) +
  geom_col(show.legend = FALSE) +
  coord_flip() +
  scale_fill_manual(values = seg_colors) +
  scale_y_continuous(labels = dollar_format(prefix = "£")) +
  labs(title = "Total Revenue by RFM Segment", x = NULL, y = "Revenue") +
  theme_retail
print(p_seg2)

p_seg3 <- df_rfm %>%
  filter(Monetary < quantile(Monetary, 0.99)) %>%
  ggplot(aes(x = Recency, y = Monetary, color = Segment)) +
  geom_point(alpha = 0.4, size = 1.5) +
  scale_color_manual(values = seg_colors) +
  scale_y_continuous(labels = dollar_format(prefix = "£")) +
  labs(title = "Recency vs Monetary Value",
       x = "Recency (days)", y = "Total Revenue") +
  theme_retail +
  theme(legend.position = "bottom")
print(p_seg3)

p_seg4 <- df_rfm %>%
  group_by(R_Score, F_Score) %>%
  summarise(Avg_M = mean(Monetary), .groups = "drop") %>%
  ggplot(aes(x = factor(F_Score), y = factor(R_Score), fill = Avg_M)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = dollar(round(Avg_M, 0), prefix = "£")), size = 3) +
  scale_fill_gradient(low = "#fff7bc", high = "#d73027",
                      labels = dollar_format(prefix = "£")) +
  labs(title = "RFM Heatmap: Avg Monetary Value",
       x = "Frequency Score", y = "Recency Score", fill = "Avg Revenue") +
  theme_retail
print(p_seg4)


cluster_data <- df_rfm %>%
  select(Recency, Frequency, Monetary, AvgOrderValue) %>%
  mutate(CLV_proxy = Frequency * Monetary)   # rough CLV

cluster_scaled <- scale(cluster_data)
set.seed(42)
wss <- map_dbl(1:10, ~ kmeans(cluster_scaled, centers = .x, nstart = 25)$tot.withinss)
elbow_df <- tibble(k = 1:10, wss = wss)
p_elbow <- ggplot(elbow_df, aes(x = k, y = wss)) +
  geom_line(color = "#2E86AB", linewidth = 1.2) +
  geom_point(size = 3, color = "#E84855") +
  scale_x_continuous(breaks = 1:10) +
  labs(title = "Elbow Method — Optimal Number of Clusters",
       x = "Number of Clusters (k)", y = "Total Within-Cluster SS") +
  theme_retail
print(p_elbow)


sil_scores <- map_dbl(2:8, function(k) {
  km <- kmeans(cluster_scaled, centers = k, nstart = 25)
  mean(silhouette(km$cluster, dist(cluster_scaled))[, 3])
})
cat("Silhouette scores for k=2 to 8:\n")
print(round(sil_scores, 3))
# ── Fit K-Means (use k=4 or whatever elbow/silhouette suggests) ──────────────────
k_opt <- 4   # adjust after seeing elbow plot
set.seed(42)
km_model <- kmeans(cluster_scaled, centers = k_opt, nstart = 25, iter.max = 100)

df_rfm$KMeans_Cluster <- factor(km_model$cluster)
