# ============================================
# app.R
# NCRB Crime Analysis Dashboard (Final Dataset)
# ============================================

library(shiny)
library(tidyverse)
library(ggplot2)

# --------------------------------------------
# Load cleaned crime dataset
# --------------------------------------------
crime <- read_csv("data/cleaned_crime_data.csv", show_col_types = FALSE)

# --------------------------------------------
# UI
# --------------------------------------------
ui <- fluidPage(
  
  titlePanel("India Crime Analysis Dashboard (NCRB Data)"),
  
  sidebarLayout(
    sidebarPanel(
      
      selectInput(
        "state",
        "Select State / UT:",
        choices = sort(unique(crime$`STATE/UT`)),
        selected = unique(crime$`STATE/UT`)[1]
      ),
      
      selectInput(
        "crime_type",
        "Select Crime Type:",
        choices = c("All", sort(unique(crime$crime_type))),
        selected = "All"
      ),
      
      sliderInput(
        "year",
        "Select Year Range:",
        min = min(crime$YEAR),
        max = max(crime$YEAR),
        value = c(min(crime$YEAR), max(crime$YEAR)),
        sep = ""
      )
    ),
    
    mainPanel(
      tabsetPanel(
        
        tabPanel(
          "Overview",
          h4("Year-wise Crime Trend (India)"),
          plotOutput("yearPlot", height = "350px"),
          br(),
          h4("Top Crime Types"),
          plotOutput("crimeTypePlot", height = "350px")
        ),
        
        tabPanel(
          "State-wise Analysis",
          h4("Crime Trend for Selected State"),
          plotOutput("stateTrendPlot", height = "350px"),
          br(),
          h4("Crime Distribution by Type"),
          plotOutput("stateCrimeTypePlot", height = "350px")
        )
      )
    )
  )
)

# --------------------------------------------
# Server
# --------------------------------------------
server <- function(input, output) {
  
  # Reactive filtered dataset
  filtered_data <- reactive({
    
    data <- crime %>%
      filter(
        `STATE/UT` == input$state,
        YEAR >= input$year[1],
        YEAR <= input$year[2]
      )
    
    if (input$crime_type != "All") {
      data <- data %>% filter(crime_type == input$crime_type)
    }
    
    data
  })
  
  # ------------------------------------------
  # 1. Year-wise crime trend (India)
  # ------------------------------------------
  output$yearPlot <- renderPlot({
    
    crime %>%
      filter(
        YEAR >= input$year[1],
        YEAR <= input$year[2]
      ) %>%
      group_by(YEAR) %>%
      summarise(total = sum(crime_count), .groups = "drop") %>%
      ggplot(aes(YEAR, total)) +
      geom_line(color = "firebrick", linewidth = 1) +
      geom_point() +
      scale_x_continuous(breaks = sort(unique(crime$YEAR))) +
      labs(
        x = "Year",
        y = "Total Crimes"
      ) +
      theme_minimal()
  })
  
  # ------------------------------------------
  # 2. Top crime types (India)
  # ------------------------------------------
  output$crimeTypePlot <- renderPlot({
    
    crime %>%
      filter(
        YEAR >= input$year[1],
        YEAR <= input$year[2]
      ) %>%
      group_by(crime_type) %>%
      summarise(total = sum(crime_count), .groups = "drop") %>%
      arrange(desc(total)) %>%
      slice_head(n = 10) %>%
      ggplot(aes(reorder(crime_type, total), total)) +
      geom_col(fill = "darkorange") +
      coord_flip() +
      labs(
        x = "Crime Type",
        y = "Total Crimes"
      ) +
      theme_minimal()
  })
  
  # ------------------------------------------
  # 3. State-wise crime trend
  # ------------------------------------------
  output$stateTrendPlot <- renderPlot({
    
    filtered_data() %>%
      group_by(YEAR) %>%
      summarise(total = sum(crime_count), .groups = "drop") %>%
      ggplot(aes(YEAR, total)) +
      geom_line(color = "steelblue", linewidth = 1) +
      geom_point() +
      scale_x_continuous(breaks = sort(unique(crime$YEAR))) +
      labs(
        x = "Year",
        y = "Total Crimes"
      ) +
      theme_minimal()
  })
  
  # ------------------------------------------
  # 4. Crime type distribution (state)
  # ------------------------------------------
  output$stateCrimeTypePlot <- renderPlot({
    
    filtered_data() %>%
      group_by(crime_type) %>%
      summarise(total = sum(crime_count), .groups = "drop") %>%
      arrange(desc(total)) %>%
      slice_head(n = 10) %>%
      ggplot(aes(reorder(crime_type, total), total)) +
      geom_col(fill = "seagreen") +
      coord_flip() +
      labs(
        x = "Crime Type",
        y = "Total Crimes"
      ) +
      theme_minimal()
  })
}

# --------------------------------------------
# Run App
# --------------------------------------------
shinyApp(ui = ui, server = server)
