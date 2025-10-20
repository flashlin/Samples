using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class BetBuilderLegEntityConfiguration : IEntityTypeConfiguration<BetBuilderLegEntity>
    {
        public void Configure(EntityTypeBuilder<BetBuilderLegEntity> builder)
        {
            builder.ToTable("BetBuilderLeg");

            builder.HasKey(x => x.Id);

            builder.Property(x => x.Id)
                .HasColumnType("bigint(19,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.TransId)
                .HasColumnType("bigint(19,0)")
                .IsRequired()
            ;

            builder.Property(x => x.RefNo)
                .HasColumnType("bigint(19,0)")
                .IsRequired()
            ;

            builder.Property(x => x.MatchMarketSelectionId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.Status)
                .HasColumnType("nvarchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.MarketTypeId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.SelectionTypeId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.Point)
                .HasColumnType("decimal(12,2)")
            ;

            builder.Property(x => x.MatchMarketDetails)
                .HasColumnType("nvarchar(4000)")
                .HasMaxLength(4000)
            ;

            builder.Property(x => x.MatchSelectionDetails)
                .HasColumnType("nvarchar(4000)")
                .HasMaxLength(4000)
            ;

            builder.Property(x => x.CustomerId)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.TransDate)
                .HasColumnType("datetime")
                .IsRequired()
            ;

            builder.Property(x => x.Ruben)
                .HasColumnType("bit")
            ;

            builder.Property(x => x.LastModifiedOn)
                .HasColumnType("datetime")
                .IsRequired()
            ;

        }
    }
}
