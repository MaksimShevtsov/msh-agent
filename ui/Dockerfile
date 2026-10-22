FROM node:22-bookworm-slim AS build
WORKDIR /ui
COPY package.json package-lock.json* ./
RUN npm install --no-audit --no-fund
COPY . .
RUN npm run build

FROM node:22-bookworm-slim
WORKDIR /ui
ENV NODE_ENV=production PORT=3000 HOSTNAME=0.0.0.0
COPY --from=build /ui/.next/standalone ./
COPY --from=build /ui/.next/static ./.next/static
EXPOSE 3000
CMD ["node", "server.js"]
